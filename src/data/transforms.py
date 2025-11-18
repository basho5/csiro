import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Tuple


def _wrap_alb(compose):
    """Return a callable that accepts a PIL Image and returns a tensor-like object.

    Albumentations expects a numpy array and returns a dict; CsiroDataset calls
    transforms(image) with a PIL image, so we wrap the compose here.
    """

    def fn(img):
        return compose(image=np.array(img))["image"]

    return fn


def get_transforms(version: int, img_size: int, finetuning: bool = True) -> Tuple[callable, callable, str]:
    if finetuning:
        normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0)
    else:
        normalize = A.Normalize(mean=[0.702, 0.522, 0.416], std=[0.139, 0.131, 0.123], max_pixel_value=255.0, p=1.0)

    if version == 1:
        transforms_train = A.Compose(
            [
                A.Resize(img_size, img_size),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                normalize,
                ToTensorV2(),
            ],
            p=1.0,
        )
        transforms_test = A.Compose([A.Resize(img_size, img_size), normalize, ToTensorV2()], p=1.0)
        t_type = "albumentations"
    else:
        # fallback to simple resize + normalize
        transforms_train = A.Compose([A.Resize(img_size, img_size), normalize, ToTensorV2()], p=1.0)
        transforms_test = A.Compose([A.Resize(img_size, img_size), normalize, ToTensorV2()], p=1.0)
        t_type = "albumentations"

    return _wrap_alb(transforms_train), _wrap_alb(transforms_test), t_type
