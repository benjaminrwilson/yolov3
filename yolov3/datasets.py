import os

import cv2
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets.coco import CocoDetection
from localization.bboxes import BBoxes, CoordType

from yolov3 import utils


class COCODataset(CocoDetection):
    def __init__(self, ann_file, root, size, ids=None):
        super(COCODataset, self).__init__(root, ann_file)

        self.ids = sorted(self.ids)

        self.coco_full_to_coco = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.coco_to_coco_full = {
            v: k for k, v in self.coco_full_to_coco.items()
        }
        self.size = size

    def __getitem__(self, index):
        img, anns = super(COCODataset, self).__getitem__(index)

        anns = [ann for ann in anns if not ann["iscrowd"]]

        w, h = img.size
        bboxes = [ann["bbox"] for ann in anns]
        target = BBoxes(Tensor(bboxes), CoordType.XYXY, (w, h))

        labels = [ann["category_id"] for ann in anns]
        target.attrs["labels"] = Tensor(labels)

        img, w, h, dw, dh = utils.transform_input(img, self.size)
        return img, target, index, w, h, dw, dh


def collate_fn(batch):
    imgs, bboxes, indices, w, h, dw, dh = zip(*batch)
    return torch.stack((imgs)), bboxes, Tensor(indices), w, h, dw, dh


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
