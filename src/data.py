import os

import cv2
import numpy as np
import torch
import utils
from torch.utils.data.dataset import Dataset


class YoloDataset(Dataset):
    def __init__(self, img_path, img_size):
        self.img_path = img_path
        self.img_size = img_size
        self.img_names = os.listdir(img_path)

    def __getitem__(self, index):
        img_name = os.path.join(self.img_path, self.img_names[index])
        img = cv2.imread(img_name)

        img = utils.transform_input(img, self.img_size)
        return img, img_name

    def __len__(self):
        return len(self.img_names)
