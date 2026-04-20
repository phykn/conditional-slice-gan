import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import cv2
import albumentations as A


class ImageDataset:
    def __init__(
        self,
        image_path: str,
        image_size: int = 64,
    ):
        self.image_path = image_path
        self.image_size = image_size

        self.image = self.read_data()
        self.build_transform()

    def __len__(self):
        return 512

    def read_data(self):
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def build_transform(self):
        self.transform = A.Compose(
            [
                A.RandomCrop(height=self.image_size, width=self.image_size),
                A.RandomRotate90(p=1.0),
                A.VerticalFlip(p=0.5),
            ]
        )

    @staticmethod
    def normalize(image):
        return (image / 127.5) - 1.0

    def __getitem__(self, idx):
        image = self.transform(image=self.image)["image"]
        image = self.normalize(image)
        image = image[None, :, :]
        return image
