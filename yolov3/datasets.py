import os

import cv2
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets.coco import CocoDetection
from localization.bboxes import BBoxes, CoordType

from yolov3 import utils
import numpy as np


class COCODataset(CocoDetection):
    def __init__(self, ann_file, root, size, ids=None):
        super(COCODataset, self).__init__(root, ann_file)

        self.ids = sorted(ids)

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
        return img, target, self.ids[index], w, h, dw, dh


def collate_fn(batch):
    imgs, bboxes, indices, w, h, dw, dh = zip(*batch)
    return torch.stack((imgs)), bboxes, Tensor(indices), w, h, dw, dh
