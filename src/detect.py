import argparse

import cv2
import numpy as np
import torch
import utils

import models
from data import YoloDataset
from torchvision import transforms


def test(opts):
    # Create model
    model = models.Darknet(opts.cfg, opts.weights, opts.nms, opts.obj)

    data_transform = transforms.Compose([
        transforms.ToTensor()])
    dataset = YoloDataset(opts.src, opts.size, data_transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1, shuffle=True,
                                             num_workers=4)
    model.eval()
    for (img, img_name) in dataloader:
        with torch.no_grad():
            detections = model(img)
            detections = detections.view(-1, 7)
            utils.write_detections(detections, img_name[0], opts.size, opts.dst)



def main():
    opts = argparse.ArgumentParser(description='Yolov3 Detection')
    opts.add_argument('-c', '--cfg', help='Configuration path',
                      default="../cfg/yolov3.cfg")
    opts.add_argument('-w', '--weights', help='Weights path',
                      default="../weights/yolov3.weights")
    opts.add_argument('-o', '--obj', help='Objectness threshold', default=.5)
    opts.add_argument(
        '-n', '--nms', help='Non-maximum Supression threshold', default=.4)
    opts.add_argument(
        '-s', '--size', help='Input size', default=608)
    opts.add_argument(
        '-src', '--src', help='Source directory', default="../images")
    opts.add_argument(
        '-d', '--dst', help='Destination directory', default="../results")
    opts = opts.parse_args()
    test(opts)


if __name__ == "__main__":
    main()
