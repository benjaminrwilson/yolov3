import argparse
import os
import time

import cv2
import torch
from torch import optim

from yolov3.src import utils
from yolov3.src.data import YoloDataset
from yolov3.src.get_image_size import get_image_size
from yolov3.src.losses import YoloLoss
from yolov3.src import models


def test(opts):
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = YoloDataset(opts.src, opts.size)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=True,
                                             num_workers=4)

    training = False
    if opts.mode == "training":
        training = True

    model = models.Darknet(opts.cfg, opts.weights,
                           opts.nms, opts.obj,
                           opts.size, device,
                           training).to(device)

    if not training:
        model.eval()

    class_colors = utils.generate_class_colors(model.num_classes)
    class_to_names = utils.get_class_names(opts.names_path)
    with torch.no_grad():
        if opts.mode == "images":
            run_detect(model, dataloader, opts,
                       class_colors, class_to_names, device)
        elif opts.mode == "cam":
            run_cam_detect(model,
                           opts,
                           class_colors,
                           class_to_names,
                           device)
        elif opts.mode == "training":
            run_train(model, dataloader, opts, device)
        elif opts.mode == "map":
            run_detect(model,
                       dataloader,
                       opts,
                       class_colors,
                       class_to_names,
                       device,
                       True)


def run_detect(model, dataloader, opts, class_colors,
               class_to_names,
               device,
               use_map=False):
    results = {}
    for i, (img, img_name) in enumerate(dataloader):
        start_time = time.time()
        img = img.to(device)
        detections = model(img)
        if len(detections) > 0:
            detections = detections.view(-1, 7)
            abs_img_path = os.path.join(opts.src, img_name[0])
            width, height = get_image_size(abs_img_path)
            detections = utils.transform_detections(
                detections, width, height, opts.size)
        if use_map:
            results[img_name[0]] = detections
        else:
            utils.write_detections(
                detections,
                img_name[0],
                opts.size,
                opts.dst,
                class_colors,
                class_to_names)
        elapsed = round(time.time() - start_time, 2)
        info = "Processed image {} in {} seconds".format(i, elapsed)
        print(info)
        # if i == 10:
        #     break
        break

    if use_map:
        print("Not Implemented")
        return
        # utils.calculate_map(opts.ann_path, opts.src, results, device,
        #                     class_colors, class_to_names, opts.size, opts.dst)


def run_cam_detect(model, opts, class_colors, class_to_names, device):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            start_time = time.time()
            img = utils.transform_input(
                frame, opts.size).unsqueeze(0).to(device)
            detections = model(img)
            print(detections)

            utils.write_detections_cam(
                detections, frame, opts.size, class_colors, class_to_names)
            utils.write_fps(frame, start_time)

            cv2.imshow('img', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def run_map(model, dataloader, opts):
    pass


def run_train(model, dataloader, opts, device):
    criterion = YoloLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    for i, (img, img_name) in enumerate(dataloader):
        start_time = time.time()
        img = img.to(device)
        ann_name = img_name[0].split("/")[-1].split(".")[0] + ".txt"
        abs_img_path = os.path.join(opts.src, img_name[0])

        frame = cv2.imread(abs_img_path)
        targets = utils.get_targets(opts, ann_name, img_name[0])
        print(targets.shape)

        loss = model(img, targets, frame)
        return


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


def main():
    opts = argparse.ArgumentParser(description='Yolov3 Detection')
    opts.add_argument('-c', '--cfg', help='Configuration path',
                      default="../cfg/yolov3.cfg")
    opts.add_argument('-w', '--weights', help='Weights path',
                      default="../weights/yolov3.weights")
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
        '-m', '--mode', help='Use video camera for demo', default="cam")
    opts.add_argument(
        '-np', '--names_path', help='Path to names of classes',
        default="../cfg/coco.names")
    opts.add_argument(
        '-a', '--ann_path', help='Path to annotations of the images',
        default="../annotations/")
    opts = opts.parse_args()
    test(opts)


if __name__ == "__main__":
    main()
