"""沿用原项目 Albumentations 1.x 的增强与两路同步翻转约定。"""

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def get_transforms(config, training=False):
    size = config["img_size"]
    normalize = [A.Normalize(mean=config["mean"], std=config["std"]), ToTensorV2()]
    resize = A.Resize(size, size, interpolation=cv2.INTER_LINEAR_EXACT)
    if not training:
        return A.Compose([resize, *normalize])

    def train(satellite):
        return A.Compose([
            A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
            A.Resize(size, size, interpolation=cv2.INTER_LINEAR_EXACT),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=0.5),
            A.OneOf([A.AdvancedBlur(p=1), A.Sharpen(p=1)], p=0.3),
            A.OneOf([
                A.GridDropout(ratio=0.4, p=1),
                A.CoarseDropout(max_holes=25, min_holes=10,
                                max_height=max(1, int(0.2 * size)), max_width=max(1, int(0.2 * size)),
                                min_height=max(1, int(0.1 * size)), min_width=max(1, int(0.1 * size)), p=1),
            ], p=0.3),
            *([A.RandomRotate90(p=1)] if satellite else []),
            A.Normalize(mean=config["mean"], std=config["std"]), ToTensorV2(),
        ])
    return train(True), train(False)
