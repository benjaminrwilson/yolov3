import argparse
import os
import time

import cv2
import torch

from yolov3.src import models, utils


def run(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.Darknet(opts.cfg, opts.weights,
                           opts.nms, opts.obj,
                           opts.size, device,
                           False).to(device).eval()

    class_colors = utils.generate_class_colors(model.num_classes)
    class_to_names = utils.get_class_names(opts.names_path)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            start_time = time.time()
            detections = model.detect(frame)
            utils.write_detections_cam(
                detections, frame, opts.size, class_colors, class_to_names)
            utils.write_fps(frame, start_time)

            cv2.imshow('img', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def get_args():
    home = os.path.expanduser("~")
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
        default="../config/coco.names")
    opts.add_argument(
        '-a', '--ann_path', help='Path to annotations of the images',
        default="../annotations/")
    opts = opts.parse_args()
    return opts


def main():
    opts = get_args()
    run(opts)


if __name__ == "__main__":
    main()
