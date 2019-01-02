import argparse
import json
import os

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from yolov3.src.datasets import COCODataset
from yolov3.src.models import Darknet


def eval(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opts.cfg, opts.weights,
                    opts.nms, opts.obj,
                    opts.size, device,
                    False).to(device).eval()

    coco_ids = get_coco_ids("../data/5k.txt")
    coco_dataset = COCODataset(opts.ann_file, opts.root, coco_ids)

    results = []
    for i, (img, target, idx) in enumerate(tqdm(coco_dataset)):
        coco_id = coco_dataset.ids[idx]
        if coco_id in coco_ids:
            detections = model.detect(rgb2bgr(img))
            if detections.shape[0] > 0:
                results += convert_to_coco_results(
                    detections, coco_id, coco_dataset)
    results = np.array(results)
    evaluate_coco(opts.ann_file, coco_ids, results)


def rgb2bgr(pil_img):
    return np.array(pil_img)[:, :, ::-1].copy()


def evaluate_coco(ann_file, ids, results):
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(results)
    ann_ids = coco_gt.getAnnIds(ids)
    coco_gt = coco_gt.loadRes(coco_gt.loadAnns(ann_ids))

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def get_coco_ids(split_file):
    coco_ids = set()
    with open(split_file, "r") as img_file:
        lines = img_file.read().splitlines()
        for line in lines:
            file_id = line.split("_")[-1].replace(".jpg", "")
            file_id = int(file_id)
            coco_ids.add(file_id)
    return sorted(list(coco_ids))


def convert_to_coco_results(detections, coco_id, coco_dataset):
    for i in range(detections.shape[0]):
        idx = coco_dataset.coco_to_coco_full[int(detections[i, 0].item())]
        detections[i, 0] = torch.Tensor([idx])

    widths = detections[:, 4] - detections[:, 2]
    heights = detections[:, 5] - detections[:, 3]

    detections[:, 4] = widths
    detections[:, 5] = heights
    detections = torch.clamp(detections, min=0)

    index = torch.LongTensor([2, 3, 4, 5, 1, 0])
    detections = detections[:, index].cpu()
    id_column = torch.zeros([detections.shape[0], 1]).fill_(coco_id)
    detections = torch.cat((id_column, detections), dim=1)
    return detections.numpy().tolist()


def get_args():
    home = os.path.expanduser("~")
    weights_file = os.path.join(home, ".torch/yolov3/yolov3.weights")
    opts = argparse.ArgumentParser(description='Yolov3 Detection')
    opts.add_argument('-c', '--cfg', help='Configuration file',
                      default="../config/yolov3.cfg")
    opts.add_argument('-w', '--weights', help='Weights file',
                      default=weights_file)
    opts.add_argument('-o', '--obj', help='Objectness threshold', default=.005)
    opts.add_argument(
        '-n', '--nms', help='Non-maximum Suppression threshold', default=.45)
    opts.add_argument(
        '-s', '--size', help='Input size', default=416)
    opts.add_argument(
        '-r', '--root', help='Root directory of images',
        default=os.path.join(home, "data/coco/val2014/"))
    opts.add_argument(
        '-d', '--dst', help='Destination directory', default="../results")
    opts.add_argument(
        '-np', '--names_path', help='Path to names of classes',
        default="../config/coco.names")
    opts.add_argument(
        '-a', '--ann_file', help='Absolute path to the coco json file',
        default=os.path.join(home, "data/coco/annotations/instances_val2014.json"))
    opts = opts.parse_args()
    return opts


def main():
    opts = get_args()
    eval(opts)


if __name__ == "__main__":
    main()
