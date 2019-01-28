import argparse
import os
import time

import cv2
import torch

from yolov3 import models, utils


def run(opts):
    assert(opts.size % 32 == 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.Darknet(opts.cfg, opts.weights,
                           opts.nms, opts.obj,
                           opts.size, device,
                           False).to(device).eval()

    class_colors = utils.generate_class_colors(model.num_classes)
    class_to_names = utils.get_class_names(opts.names_path)

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if ret:
            bboxes = model.detect(img)
            annotated_img = utils.show_bboxes(bboxes, img)

            cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def get_args():
    home = os.path.expanduser("~")
    weights_file = os.path.join(home, ".torch/yolov3/yolov3.weights")
    opts = argparse.ArgumentParser(description='Yolov3 Detection')
    opts.add_argument('-c', '--cfg',
                      help='Configuration file',
                      default="../configs/yolov3.cfg")
    opts.add_argument('-w', '--weights',
                      help='Weights file',
                      default=weights_file)
    opts.add_argument('-o', '--obj',
                      help='Objectness threshold',
                      default=.5)
    opts.add_argument('-n', '--nms',
                      help='Non-maximum Suppression threshold',
                      default=.45)
    opts.add_argument('-s', '--size',
                      help='Input size',
                      default=320)
    opts.add_argument('-np', '--names_path',
                      help='Path to names of classes',
                      default="../data/coco.names")
    opts = opts.parse_args()
    return opts


def main():
    opts = get_args()
    run(opts)


if __name__ == "__main__":
    main()
