from __future__ import annotations

import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import albumentations as A
import cv2
import numpy as np
from imrw import imread


class SliceDataset:
    def __init__(
        self,
        image_path: str,
        image_size: int = 64,
        in_channels: int = 1,
        steps_per_epoch: int = 512,
    ) -> None:
        assert in_channels in (1, 3), "in_channels must be 1 or 3"
        self.image_path = image_path
        self.image_size = image_size
        self.in_channels = in_channels
        self.steps_per_epoch = steps_per_epoch

        self.image = self._read()
        self.transform = A.Compose(
            [
                A.RandomCrop(height=image_size, width=image_size),
                A.RandomRotate90(p=1.0),
                A.VerticalFlip(p=0.5),
            ]
        )

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, idx: int) -> np.ndarray:
        image = self.transform(image=self.image)["image"]
        image = (image.astype(np.float32) / 127.5) - 1.0
        if self.in_channels == 1:
            image = image[None, :, :]
        else:
            image = np.transpose(image, (2, 0, 1))
        return np.ascontiguousarray(image)

    def _read(self) -> np.ndarray:
        image = imread(self.image_path)
        if self.in_channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
