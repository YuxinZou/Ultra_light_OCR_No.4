# Modified from mmlab
import albumentations as albu
from albumentations import Compose
import imgaug.augmenters as iaa


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
