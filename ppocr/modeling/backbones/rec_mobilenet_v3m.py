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

import paddle
from paddle import nn
from paddle import ParamAttr
from paddle.nn import functional as F

__all__ = ['MobileNetV3M']


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV3M(nn.Layer):
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
        overwrite_act=None,
        force_shortcut=False,
        force_se=False,
        act_residual=False,
        **kwargs,
    ):
        super(MobileNetV3M, self).__init__()
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
            groups=1,
            if_act=True,
            act=overwrite_act if overwrite_act else 'hardswish',
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
                    use_se=True if force_se else se,
                    act=overwrite_act if overwrite_act else nl,
                    se_act=overwrite_act if overwrite_act else 'default',
                    force_shortcut=force_shortcut,
                    act_last=act_residual,
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
            groups=1,
            if_act=True,
            act=overwrite_act if overwrite_act else 'hardswish',
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


class ResidualUnit(nn.Layer):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        kernel_size,
        stride,
        use_se,
        act=None,
        force_shortcut=False,
        act_last=False,
        se_act='default',
        name='',
    ):
        super(ResidualUnit, self).__init__()

        self.act_last = act_last

        if force_shortcut:
            self.if_shortcut = True
            short_list = []
            if (
                (isinstance(stride, tuple) and max(stride) > 1) or
                (not isinstance(stride, tuple) and stride > 1)
            ):
                short_list.append(nn.AvgPool2D(
                    kernel_size=stride,
                    stride=stride,
                    padding=0,
                    ceil_mode=True,
                ))
            if in_channels != out_channels:
                short_list.append(ConvBNLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    if_act=False,
                    name=name + '_short_conv',
                ))
            if short_list:
                self.short = nn.Sequential(*short_list)
            else:
                self.short = None
        else:
            self.if_shortcut = stride == 1 and in_channels == out_channels
            self.short = None
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            if_act=True,
            act=act,
            name=name + "_expand",
        )
        self.bottleneck_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=mid_channels,
            if_act=True,
            act=act,
            name=name + "_depthwise",
        )
        if self.if_se:
            self.mid_se = SEModule(
                mid_channels,
                name=name + "_se",
                act=se_act,
            )
        self.linear_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            if_act=False,
            act=None,
            name=name + "_linear",
        )

        if act_last:
            if act == 'relu':
                self.act1 = nn.ReLU()
            elif act == 'hardswish':
                self.act1 = nn.Hardswish()
            elif act == 'swish':
                self.act1 = nn.Swish()
            else:
                raise ValueError(f'act layer: {act} is not yet supported!')

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            if self.short is not None:
                short = self.short(inputs)
            else:
                short = inputs
            x = paddle.add(short, x)
        if self.act_last:
            x = self.act1(x)

        return x


class SEModule(nn.Layer):
    def __init__(
        self,
        in_channels,
        reduction=4,
        name="",
        act='default',
    ):
        super(SEModule, self).__init__()
        self.act = act
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(name=name + "_1_weights"),
            bias_attr=ParamAttr(name=name + "_1_offset"),
        )
        self.conv2 = nn.Conv2D(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(name + "_2_weights"),
            bias_attr=ParamAttr(name=name + "_2_offset"),
        )
        if act == 'default':
            self.act1 = nn.ReLU()
            self.act2 = nn.Hardsigmoid()
        else:
            if act == 'swish':
                self.act1 = nn.Swish()
            else:
                raise ValueError(f'act layer: {act} is not yet supported!')
            self.act2 = nn.Sigmoid()

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = self.act1(outputs)
        outputs = self.conv2(outputs)
        if self.act == 'default':
            # because stupid paddle doesn't support forward kwargs
            outputs = F.hardsigmoid(outputs, slope=0.2, offset=0.5)
        else:
            self.act2(outputs)

        return inputs * outputs


class ConvBNLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        groups=1,
        if_act=True,
        act=None,
        name=None,
    ):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + '_weights'),
            bias_attr=False,
        )

        self.norm = nn.BatchNorm(
            num_channels=out_channels,
            act=None,
            param_attr=ParamAttr(name=name + "_bn_scale"),
            bias_attr=ParamAttr(name=name + "_bn_offset"),
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance",
        )

        if self.if_act:
            if act == 'relu':
                self.act = nn.ReLU()
            elif act == 'hardswish':
                self.act = nn.Hardswish()
            elif act == 'swish':
                self.act = nn.Swish()
            else:
                raise ValueError(f'act layer: {act} is not yet supported!')
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.act is not None:
            x = self.act(x)

        return x
