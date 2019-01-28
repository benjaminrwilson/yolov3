import argparse
import json
import os

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from localization.bboxes import BBoxes, CoordType
from yolov3.datasets import COCODataset, collate_fn
from yolov3.models import Darknet
from yolov3.utils import transform_detections


def eval(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opts.cfg, opts.weights,
                    opts.nms, opts.obj,
                    opts.size, device,
                    False).to(device).eval()

    coco_ids = get_coco_ids("../data/5k.txt")
    coco_dataset = COCODataset(opts.ann_file, opts.root, opts.size, coco_ids)

    dataloader = torch.utils.data.DataLoader(coco_dataset,
                                             batch_size=16,
                                             collate_fn=collate_fn,
                                             num_workers=4,
                                             shuffle=True)

    results = []
    for i, (img, target, ids, w, h, dw, dh) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            dets = model.forward(img.to(device))
            bboxes = transform_detections(dets, w, h, dw, dh, opts.size)
            results += convert_to_coco_results(
                bboxes, ids, coco_dataset, device)
            if i > 10:
                break
    results = np.array(results)
    evaluate_coco(opts.ann_file, coco_ids, results)


def evaluate_coco(ann_file, ids, results):
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(results)
    ann_ids = coco_gt.getAnnIds(ids)
    print(ann_ids)
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


def convert_to_coco_results(batches, ids, coco_dataset, device):
    results = []
    used_ids = {x: i for i, x in enumerate(list(sorted(ids.numpy())))}
    print(used_ids)
    for i, bboxes in enumerate(batches):
        labels = bboxes.attrs["labels"].unsqueeze(1)
        coco_id = ids[i]
        print(coco_id)

        n = labels.shape[0]
        for i in range(n):
            idx = coco_dataset.coco_to_coco_full[int(labels[i].item())]
            labels[i].fill_(idx)

        coco_ids = torch.zeros([n]).fill_(coco_id).to(device).unsqueeze(1)
        coords = bboxes.convert(CoordType.XYWH).coords
        confidences = bboxes.attrs["confidences"].unsqueeze(1)
        res = torch.cat((coco_ids, coords, confidences, labels), dim=-1)
        res = res.cpu().numpy().tolist()
        results += res
    return results


def get_args():
    home = os.path.expanduser("~")
    weights_file = os.path.join(home, ".torch/yolov3/yolov3.weights")
    opts = argparse.ArgumentParser(description='Yolov3 Evaluation')
    opts.add_argument('-c', '--cfg',
                      help='Configuration file',
                      default="../configs/yolov3.cfg")
    opts.add_argument('-w', '--weights',
                      help='Weights file',
                      default=weights_file)
    opts.add_argument('-o', '--obj',
                      help='Objectness threshold',
                      default=.005)
    opts.add_argument('-n', '--nms',
                      help='Non-maximum Suppression threshold',
                      default=.45)
    opts.add_argument('-s', '--size',
                      help='Input size', default=608)
    opts.add_argument('-r', '--root',
                      help='Root directory of images',
                      default="../datasets/coco/val2014/")
    opts.add_argument('-d', '--dst',
                      help='Destination directory',
                      default="../results")
    opts.add_argument('-np', '--names_path',
                      help='Path to names of classes',
                      default="../config/coco.names")
    opts.add_argument('-a', '--ann_file',
                      help='Absolute path to the coco json file',
                      default="../datasets/coco/annotations/"
                      "instances_val2014.json")
    opts = opts.parse_args()
    return opts


def main():
    opts = get_args()
    eval(opts)


if __name__ == "__main__":
    main()
