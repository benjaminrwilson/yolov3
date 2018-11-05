import argparse

import cv2
import numpy as np
import torch
import utils

import models
from data import YoloDataset


def test(opts):
    # Create model
    model = models.Darknet(opts.cfg, opts.weights,
                           opts.nms, opts.obj, opts.size)

    dataset = YoloDataset(opts.src, opts.size)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1, shuffle=True,
                                             num_workers=4)
    model.eval()

    class_colors = utils.generate_class_colors(model.num_classes)
    class_to_names = utils.get_class_names(opts.names_path)
    with torch.no_grad():
        if not opts.use_cam:
            run_detect(model, dataloader, opts, class_colors, class_to_names)
        else:
            run_cam_detect(model,
                           opts,
                           class_colors,
                           class_to_names)


def run_detect(model, dataloader, opts, class_colors, class_to_names):
    for (img, img_name) in dataloader:
        detections = model(img)
        detections = detections.view(-1, 7)
        utils.write_detections(
            detections,
            img_name[0],
            opts.size,
            opts.dst,
            class_colors,
            class_to_names)


def run_cam_detect(model, opts, class_colors, class_to_names, show_fps):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            img = utils.transform_input(frame, opts.size).unsqueeze(0)
            detections = model(img)
            utils.write_detections_cam(
                detections, frame, opts.size, class_colors, class_to_names)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


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
        '-s', '--size', help='Input size', default=352)
    opts.add_argument(
        '-src', '--src', help='Source directory', default="../images")
    opts.add_argument(
        '-d', '--dst', help='Destination directory', default="../results")
    opts.add_argument(
        '-uc', '--use_cam', help='Use video camera for demo', default=False)
    opts.add_argument(
        '-np', '--names_path', help='Path to names of classes',
        default="../cfg/coco.names")
    opts = opts.parse_args()
    test(opts)


if __name__ == "__main__":
    main()
