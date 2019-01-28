import os

import cv2
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets.coco import CocoDetection

from yolov3 import utils


class COCODataset(CocoDetection):
    def __init__(self, ann_file, root, ids=None):
        super(COCODataset, self).__init__(root, ann_file)

        if ids is None:
            self.ids = sorted(self.ids)
        else:
            self.ids = ids

        self.coco_full_to_coco = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.coco_to_coco_full = {
            v: k for k, v in self.coco_full_to_coco.items()
        }

    def __getitem__(self, index):
        img, anns = super(COCODataset, self).__getitem__(index)

        anns = [ann for ann in anns if not ann["iscrowd"]]

        bboxes = [ann["bbox"] for ann in anns]
        bboxes = Tensor(bboxes)

        target = [ann["category_id"] for ann in anns]
        target = Tensor(target).unsqueeze(1)
        target = torch.cat((target, bboxes), dim=1)
        return img, target, index


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
