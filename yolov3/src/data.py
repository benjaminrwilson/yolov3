import os

import cv2
from torch.utils.data.dataset import Dataset

from yolov3.src import utils


class YoloDataset(Dataset):
    def __init__(self, img_path, img_size):
        self.img_path = img_path
        self.img_size = img_size
        self.img_names = self.get_img_names(img_path)

    def __getitem__(self, index):
        img_name = os.path.join(self.img_path, self.img_names[index])
        img = cv2.imread(img_name)

        img = utils.transform_input(img, self.img_size)
        return img, img_name

    def __len__(self):
        return len(self.img_names)

    def get_img_names(self, img_path):
        img_names = []
        for img_name in os.listdir(img_path):
            if not img_name.startswith("."):
                img_names.append(img_name)
        return img_names
