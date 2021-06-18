# Modified from mmlab
import random
from typing import Optional, Union, Tuple, List, Dict

import cv2
import numpy as np
import albumentations as albu
from albumentations import Compose
import imgaug.augmenters as iaa

__all__ = [
    'Imau',
    'Albu',
    'HeightRatioCrop',
    'Resize',
    'ToFloat',
    'Normalize',
    'Pad',
    'PatchPad',
    'Transpose',
]

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


class Imau:
    """
    """
    def __init__(self, transforms, **kwargs) -> None:
        self.transforms = transforms

        self.augmenter = iaa.Sequential(
            [self.imau_builder(t) for t in self.transforms]
        )

    def imau_builder(self, cfg: dict) -> iaa.Augmenter:
        """Import a module from imgaug.
        It inherits some of :func:`build_from_cfg` logic.
        Args:
            cfg (dict): Config dict.
                It should at least contain the key "typename".
        Returns:
            obj: The constructed object.
        """
        assert isinstance(cfg, dict) and 'typename' in cfg
        args = cfg.copy()

        obj_type = args.pop('typename')
        if isinstance(obj_type, str):
            obj_cls = getattr(iaa, obj_type)
        else:
            raise TypeError(
                f'typename must be a str, but got {type(obj_type)}')

        for k in ['transforms', 'children', 'then_list', 'else_list']:
            if k in args:
                args[k] = [self.imau_builder(t) for t in args[k]]

        return obj_cls(**args)

    def __call__(self, data):
        image = data['image']
        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            data['image'] = aug.augment_image(image)

        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str


class Albu:
    """Albumentation augmentation.

    Adds custom transformations from Albumentations library.
    Please, visit `https://albumentations.readthedocs.io`
    to get more information.
    An example of ``transforms`` is as followed:

    .. code-block::
        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]

    Args:
        transforms (list[dict]): A list of albu transformations
        keymap (dict): Contains {'input key':'albumentation-style key'}
    """

    def __init__(self, transforms, **kwargs) -> None:
        self.transforms = transforms

        self.aug = Compose([self.albu_builder(t) for t in self.transforms])

    def albu_builder(self, cfg: dict) -> albu.BasicTransform:
        """Import a module from albumentations.
        It inherits some of :func:`build_from_cfg` logic.
        Args:
            cfg (dict): Config dict.
                It should at least contain the key "typename".
        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'typename' in cfg
        args = cfg.copy()

        obj_type = args.pop('typename')
        if isinstance(obj_type, str):
            obj_cls = getattr(albu, obj_type)
        else:
            raise TypeError(
                f'typename must be a str, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    def __call__(self, data):
        data = self.aug(**data)

        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str


class HeightRatioCrop:
    def __init__(
        self,
        aug_prob: float = 0.4,
        crop_ratio: Union[Tuple, List] = (0, 0.1),
        **kwargs,
    ) -> None:
        self.aug_prob = aug_prob
        self.crop_ratio = crop_ratio

    def __call__(self, data):
        img = data['image']
        h, w, _ = img.shape
        crop_range = list(map(lambda x: int(x*h), self.crop_ratio))
        crop_px = random.randint(*crop_range)
        if random.randint(0, 1):
            cropped = img[crop_px: h, ...]
        else:
            cropped = img[0: h - crop_px, ...]
        data['image'] = cropped

        return data

    def __repr__(self) -> str:
        kwargs = ', '.join([
            f'aug_prob={self.aug_prob}',
            f'crop_ratio={self.crop_ratio}',
        ])
        return f'{self.__class__.__name__}({kwargs})'


class Resize:
    """ Resize images.

    Resize image either keep ratio or keep max width, with support of
    multi-scale.

    Args:
        img_scale (tuple | list(tuple)): specifies img_scale(s) in (w, h) order
        ensures (str): "ratio" or "max_width", suggest "max_width" during train
            phase, "ratio" during test phase.
            NOTE that with "max_width" mode, images may NOT be resized w.r.t.
            image aspect ratio if it's too long.
        ms_start_epoch (bool): if True, will save "target_epoch" value to
            `data`, for Pad transform.
        ms_start_epoch (int): only do multi-scale after specified epoch num,
            otherwise use the first `img_scale`
    """
    valid_mode = ('ratio', 'max_width')

    def __init__(
        self,
        img_scale: Union[Tuple[int], List[Tuple[int]]],
        ensures: str,
        record_target_scale: bool = True,
        ms_start_epoch: int = 0,
        **kwargs,
    ) -> None:
        if isinstance(img_scale, tuple):
            img_scale = [img_scale]
        assert isinstance(img_scale, list), 'img_scale must be list of tuples!'
        assert ensures in self.valid_mode, \
            'either "ratio" or "max_width" is required for `ensures`'

        self.img_scale = img_scale
        self.ensures = ensures
        self.ms_start_epoch = ms_start_epoch
        self.record_target_scale = record_target_scale

    def select_scale(self, epoch: int, mode: str = 'train') -> Tuple[int]:
        if mode == 'train' and epoch > self.ms_start_epoch:
            return self.img_scale[epoch % len(self.img_scale)]
        else:
            if mode != 'train':
                assert len(self.img_scale) == 1, \
                    '`img_scale` must be 1 during eval/test'
            return self.img_scale[0]

    def __call__(self, data: Dict) -> Dict:
        scale = self.select_scale(epoch=data['epoch'], mode=data['mode'])
        if self.ensures == 'max_width':
            img = resize_keep_max_width(data['image'], scale)
        else:
            img = resize_keep_ratio(data['image'], scale)
        data['image'] = img
        if self.record_target_scale:
            data['target_scale'] = scale

        return data

    def __repr__(self) -> str:
        kwargs = ', '.join([
            f'img_scale={self.img_scale}',
            f'ensures={self.ensures}',
        ])
        return f'{self.__class__.__name__}({kwargs})'


class ToFloat:
    def __init__(
        self,
        key: Union[str, List[str]] = 'image',
        **kwargs,
    ) -> None:
        if isinstance(key, str):
            key = [key]
        assert isinstance(key, list), '`key` must be list or str!'
        self.key = key

    def __call__(self, data: Dict) -> Dict:
        for k in self.key:
            data[k] = np.asfarray(data[k], dtype=np.float32)

        return data

    def __repr__(self) -> str:
        kwargs = ', '.join([
            f'key={self.key}',
        ])
        return f'{self.__class__.__name__}({kwargs})'


class Normalize:
    def __init__(
        self,
        mean: List[Union[int, float]],
        std: List[Union[int, float]],
        is_rgb: bool = False,
        **kwargs,
    ) -> None:
        if is_rgb:
            mean = mean[::-1]
            std = std[::-1]
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.is_rgb = is_rgb

    def __call__(self, data: Dict) -> Dict:
        img = data['image']
        img = (img - self.mean) / self.std
        data['image'] = img

        return data

    def __repr__(self) -> str:
        kwargs = ', '.join([
            f'mean={self.mean}',
            f'std={self.std}',
            f'is_rgb={self.is_rgb}',
        ])
        return f'{self.__class__.__name__}({kwargs})'


class Pad:
    """ Pad in (h, w, c) order. """
    def __init__(
        self,
        target_scale: Optional[Tuple[int]] = None,
        use_record_target_scale: bool = True,
        **kwargs,
    ) -> None:
        assert bool(target_scale) ^ bool(use_record_target_scale), \
            'either `target_scale` or `use_record_target_scale` can be ' \
            'specified!'
        self.target_scale = target_scale
        self.use_record_target_scale = use_record_target_scale

    def __call__(self, data: Dict) -> Dict:
        orig_h, orig_w, c = data['image'].shape
        if self.use_record_target_scale:
            w, h = data['target_scale']
        else:
            w, h = self.target_scale
        canvas = np.zeros((h, w, c), dtype=data['image'].dtype)
        canvas[:orig_h, :orig_w, :] = data['image']
        data['image'] = canvas

        return data

    def __repr__(self) -> str:
        kwargs = ', '.join([
            f'target_scale={self.target_scale}',
            f'use_record_target_scale={self.use_record_target_scale}',
        ])
        return f'{self.__class__.__name__}({kwargs})'


class PatchPad:
    """ Pad in (c, h, w) order to make image size divisible.

    NOTE: Walkaround for test-time padding, do not use otherwise!
    """
    def __init__(
        self,
        divisor: int = 4,
        **kwargs,
    ) -> None:
        self.divisor = divisor

    def __call__(self, data: Dict) -> Dict:
        c, orig_h, orig_w = data['image'].shape
        h = int(np.ceil(orig_h / self.divisor) * self.divisor)
        w = int(np.ceil(orig_w / self.divisor) * self.divisor)

        canvas = np.zeros((c, h, w), dtype=np.float32)
        canvas[c, :orig_h, :orig_w] = data['image']
        data['image'] = canvas

        return data

    def __repr__(self) -> str:
        kwargs = ', '.join([
            f'divisor={self.divisor}',
        ])
        return f'{self.__class__.__name__}({kwargs})'


class Transpose:
    def __init__(
        self,
        order: List[int],
        **kwargs,
    ) -> None:
        self.order = order

    def __call__(self, data: Dict) -> Dict:
        data['image'] = data['image'].transpose(self.order)

        return data

    def __repr__(self) -> str:
        kwargs = ', '.join([
            f'order={self.order}',
        ])
        return f'{self.__class__.__name__}({kwargs})'


def resize_keep_ratio(img: np.ndarray, scale: Tuple[int]) -> np.ndarray:
    _, target_h = scale
    orig_h, orig_w, _ = img.shape
    orig_ratio = orig_w / orig_h

    # add 0.5 to ensure round
    resized_w = int(target_h*orig_ratio+0.5)
    resized_h = target_h
    resized = cv2.resize(img, (resized_w, resized_h))

    return resized


def resize_keep_max_width(img: np.ndarray, scale: Tuple[int]) -> np.ndarray:
    target_w, target_h = scale
    orig_h, orig_w, _ = img.shape
    orig_ratio = orig_w / orig_h

    # add 0.5 to ensure round
    resized_w = int(np.clip(target_h*orig_ratio+0.5, 0, target_w))
    resized_h = target_h
    resized = cv2.resize(img, (resized_w, resized_h))

    return resized
