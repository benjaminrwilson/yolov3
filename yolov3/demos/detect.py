import argparse
import os
import time
from os.path import expanduser

import cv2
import torch
from torch import optim
from tqdm import tqdm

from yolov3.src import models, utils
from yolov3.src.datasets import YoloDataset
from yolov3.src.get_image_size import get_image_size
from yolov3.src.losses import YoloLoss


def test(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = YoloDataset(opts.src, opts.size)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=True,
                                             num_workers=4)

    model = models.Darknet(opts.cfg, opts.weights,
                           opts.nms, opts.obj,
                           opts.size, device, False).to(device).eval()

    run_detect(model, dataloader, opts, device)


def run_detect(model, dataloader, opts, device):
    results = {}
    class_colors = utils.generate_class_colors(model.num_classes)
    class_to_names = utils.get_class_names(opts.names_path)
    with torch.no_grad():
        for i, (img, img_name) in enumerate(tqdm(dataloader)):
            start_time = time.time()
            img = img.to(device)
            detections = model(img)
            if detections.shape[0] > 0:
                width, height = get_image_size(img_name[0])
                detections = utils.transform_detections(
                    detections, width, height, opts.size)
                detections = detections.view(-1, 7)
                confidences = detections[..., -2] * detections[..., -3]
                detections = torch.cat((detections[..., 6].unsqueeze(1),
                                        confidences.unsqueeze(1),
                                        detections[..., :4]), 1)
        utils.write_detections(
            detections,
            img_name[0],
            opts.size,
            opts.dst,
            class_colors,
            class_to_names)
        elapsed = round(time.time() - start_time, 2)
        info = "Processed image {} in {} seconds".format(i, elapsed)


def get_model(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.Darknet(opts.cfg, opts.weights,
                           opts.nms, opts.obj,
                           opts.size, device,
                           False).to(device)
    model.eval()
    return model


class Config:
    def __init__(self, cfg, weights, nms, obj, size):
        self.cfg = cfg
        self.weights = weights
        self.nms = nms
        self.obj = obj
        self.size = size


def get_args():
    home = expanduser("~")
    weights_file = os.path.join(home, ".torch/yolov3/yolov3.weights")
    opts = argparse.ArgumentParser(description='Yolov3 Detection')
    opts.add_argument('-c', '--cfg', help='Configuration file',
                      default="../config/yolov3.cfg")
    opts.add_argument('-w', '--weights', help='Weights file',
                      default=weights_file)
    opts.add_argument('-o', '--obj', help='Objectness threshold', default=.5)
    opts.add_argument(
        '-n', '--nms', help='Non-maximum Suppression threshold', default=.45)
    opts.add_argument(
        '-s', '--size', help='Input size', default=416)
    opts.add_argument(
        '-src', '--src', help='Source directory', default="../images")
    opts.add_argument(
        '-d', '--dst', help='Destination directory', default="../results")
    opts.add_argument(
        '-np', '--names_path', help='Path to names of classes',
        default="../data/coco.names")
    opts.add_argument(
        '-a', '--ann_path', help='Path to annotations of the images',
        default="../annotations/")
    opts = opts.parse_args()
    return opts


def main():
    opts = get_args()
    test(opts)


if __name__ == "__main__":
    main()
