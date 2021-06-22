# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle import nn
from paddle.nn import functional as F

from ppocr.modeling.backbones.det_mobilenet_v3 import (
    ResidualUnit,
    ConvBNLayer,
    make_divisible,
)

__all__ = ['MobileNetV3']


class MobileNetV3(nn.Layer):
    def __init__(
        self,
        in_channels=3,
        model_name='small',
        scale=0.5,
        large_stride=None,
        small_stride=None,
        last_pool=None,
        ms_pool_type=None,
        dropout_cfg=None,
        **kwargs,
    ):
        super(MobileNetV3, self).__init__()
        if small_stride is None:
            small_stride = [2, 2, 2, 2]
        if large_stride is None:
            large_stride = [1, 2, 2, 2]

        assert isinstance(large_stride, list), \
            f"large_stride type must be list but got {type(large_stride)}"
        assert isinstance(small_stride, list), \
            f"small_stride type must be list but got {type(small_stride)}"
        assert len(large_stride) == 4, \
            f"large_stride length must be 4 but got {len(large_stride)}"
        assert len(small_stride) == 4, \
            f"small_stride length must be 4 but got {len(small_stride)}"

        if model_name == "large":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', large_stride[0]],
                [3, 64, 24, False, 'relu', (large_stride[1], 1)],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', (large_stride[2], 1)],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hardswish', 1],
                [3, 200, 80, False, 'hardswish', 1],
                [3, 184, 80, False, 'hardswish', 1],
                [3, 184, 80, False, 'hardswish', 1],
                [3, 480, 112, True, 'hardswish', 1],
                [3, 672, 112, True, 'hardswish', 1],
                [5, 672, 160, True, 'hardswish', (large_stride[3], 1)],
                [5, 960, 160, True, 'hardswish', 1],
                [5, 960, 160, True, 'hardswish', 1],
            ]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', (small_stride[0], 1)],
                [3, 72, 24, False, 'relu', (small_stride[1], 1)],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hardswish', (small_stride[2], 1)],
                [5, 240, 40, True, 'hardswish', 1],
                [5, 240, 40, True, 'hardswish', 1],
                [5, 120, 48, True, 'hardswish', 1],
                [5, 144, 48, True, 'hardswish', 1],
                [5, 288, 96, True, 'hardswish', (small_stride[3], 1)],
                [5, 576, 96, True, 'hardswish', 1],
                [5, 576, 96, True, 'hardswish', 1],
            ]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        # supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        # assert scale in supported_scale, (
        #     f'supported scales are {supported_scale} '
        #     f'but input scale is {scale}')

        inplanes = 16
        # conv1
        self.conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(inplanes * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            if_act=True,
            act='hardswish',
            name='conv1',
        )
        i = 0
        block_list = []
        inplanes = make_divisible(inplanes * scale)
        for (k, exp, c, se, nl, s) in cfg:
            block_list.append(
                ResidualUnit(
                    in_channels=inplanes,
                    mid_channels=make_divisible(scale * exp),
                    out_channels=make_divisible(scale * c),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    act=nl,
                    name='conv' + str(i + 2),
                ))
            inplanes = make_divisible(scale * c)
            i += 1
        self.blocks = nn.Sequential(*block_list)

        self.conv2 = ConvBNLayer(
            in_channels=inplanes,
            out_channels=make_divisible(scale * cls_ch_squeeze),
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            if_act=True,
            act='hardswish',
            name='conv_last',
        )

        self.multi_scale = bool(ms_pool_type)
        if self.multi_scale:
            if ms_pool_type == 'max':
                self.pool = F.adaptive_max_pool2d
            elif ms_pool_type == 'mean':
                self.pool = F.adaptive_avg_pool2d
        else:
            if last_pool is None:
                self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
            else:
                self.pool = nn.MaxPool2D(**last_pool)

        self.out_channels = make_divisible(scale * cls_ch_squeeze)

        if dropout_cfg is None:
            dropout_cfg = dict()
        self.dp_start_epoch = dropout_cfg.get('start_epoch')
        self.dp_final_p = dropout_cfg.get('final_p')
        self.dp_start_block_idx = dropout_cfg.get('start_block_idx')

    def forward(self, x, epoch=None, epoch_num=None):
        x = self.conv1(x)
        if not self.training or epoch is None or epoch < self.dp_start_epoch:
            x = self.blocks(x)
        else:
            for i, block in enumerate(self.blocks):
                x = block(x)
                if i >= self.dp_start_block_idx:
                    progress = (epoch - self.dp_start_epoch) / \
                        (epoch_num - self.dp_start_epoch)
                    x = F.dropout2d(x, p=progress*self.dp_final_p)
        x = self.conv2(x)
        if self.multi_scale:
            h, w = x.shape[-2:]
            assert w % h == 0, f'w: {w} can\'t be devided by h: {h}'
            x = self.pool(x, output_size=(1, w//h))
        else:
            x = self.pool(x)
        return x
