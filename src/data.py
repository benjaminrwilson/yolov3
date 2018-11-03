import os

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from torchvision import transforms


class YoloDataset(Dataset):
    def __init__(self, img_path, img_size, transforms=None):
        self.img_path = img_path
        self.img_size = img_size
        self.img_names = os.listdir(img_path)
        self.transforms = transforms

    def __getitem__(self, index):
        img_name = os.path.join(self.img_path, self.img_names[index])
        img = cv2.imread(img_name)

        height, width = img.shape[:2]
        if height > width:
            pad_top = int(np.ceil((height - width) / 2))
            pad_bot = (width - height) // 2
            img = cv2.copyMakeBorder(
                img, 0, 0, pad_top, pad_bot, cv2.BORDER_CONSTANT, value=(127.5, 127.5, 127.5))
        else:
            pad_top = int(np.ceil((width - height) / 2))
            pad_bot = (width - height) // 2
            img = cv2.copyMakeBorder(
                img, pad_top, pad_bot, 0, 0, cv2.BORDER_CONSTANT, value=(127.5, 127.5, 127.5))
        img = cv2.resize(img, (self.img_size, self.img_size))
        if self.transforms is not None:
            img = self.transforms(img)
        return img, img_name

    def __len__(self):
        return len(self.img_names)
