import os
import time

import cv2
import numpy as np
import torch

from torchvision import transforms


def parse_cfg(cfg_path):
    with open(cfg_path, "r") as cfg_file:
        lines = cfg_file.read().splitlines()
        lines = [l for l in lines if len(l) > 0 and l[0] != "#"]

    layers = []
    layer = {}
    for line in lines:
        if line[0] == "[":
            if layer:
                layers.append(layer)
                layer = {}
            layer["type"] = line[1:-1]
        else:
            attr, val = line.split("=")
            layer[attr.rstrip()] = val.lstrip()
    layers.append(layer)
    return layers


def darknet2corners(preds):
    bb = preds.new(preds.shape)
    bb[..., 0] = preds[..., 0] - preds[..., 2] / 2
    bb[..., 1] = preds[..., 1] - preds[..., 3] / 2
    bb[..., 2] = preds[..., 0] + preds[..., 2] / 2
    bb[..., 3] = preds[..., 1] + preds[..., 3] / 2
    preds[..., :4] = bb[..., :4]
    return preds


def bb_nms(bb1, bb2):
    bb1_x1, bb1_y1, bb1_x2, bb1_y2 = bb1[...,
                                         0], bb1[..., 1], bb1[..., 2], bb1[..., 3]
    bb2_x1, bb2_y1, bb2_x2, bb2_y2 = bb2[...,
                                         0], bb2[..., 1], bb2[..., 2], bb2[..., 3]

    intersection_x1 = torch.max(bb1_x1, bb2_x1)
    intersection_y1 = torch.max(bb1_y1, bb2_y1)
    intersection_x2 = torch.min(bb1_x2, bb2_x2)
    intersection_y2 = torch.min(bb1_y2, bb2_y2)

    interection_width = torch.clamp(intersection_x2 - intersection_x1 + 1, 0)
    interection_height = torch.clamp(intersection_y2 - intersection_y1 + 1, 0)
    intersection_area = interection_width * interection_height
    bb1_area = (bb1_x2 - bb1_x1 + 1) * (bb1_y2 - bb1_y1 + 1)
    bb2_area = (bb2_x2 - bb2_x1 + 1) * (bb2_y2 - bb2_y1 + 1)

    iou = intersection_area / (bb1_area + bb2_area - intersection_area)
    return iou


def write_detections(detections, img_path, size, dst, class_colors,
                     class_to_names):
    img = cv2.imread(img_path)
    height, width = img.shape[0:2]
    detections = transform_detections(detections, width, height, size)
    _write_detection(img, detections, class_colors, class_to_names)

    img_name = img_path.split("/")[-1]
    dst = os.path.join(dst, img_name)
    cv2.imwrite(dst, img)


def write_detections_cam(detections, img, size, class_colors, class_to_names):
    if len(detections) > 0:
        detections = detections.view(-1, 7)
        height, width = img.shape[0:2]
        detections = transform_detections(detections, width, height, size)
        _write_detection(img, detections, class_colors, class_to_names)


def _write_detection(img, detections, class_colors, class_to_names):
    for d in detections:
        x1, y1, x2, y2 = d[:4].numpy()
        class_pred = int(d[-1].numpy())
        img = cv2.rectangle(img, (x1, y1), (x2, y2),
                            class_colors[class_pred], thickness=3)
        img = cv2.putText(img, class_to_names[class_pred],
                          (x1, y1),
                          cv2.FONT_HERSHEY_DUPLEX,
                          2,
                          (255, 255, 255),
                          2)
    return img


def transform_detections(detections, width, height, size):
    ratio = max(width, height) / size
    detections[..., :4] *= ratio
    if width > height:
        pad = np.ceil((width - height) / 2)
        detections[..., 1] -= pad
        detections[..., 3] -= pad
    else:
        pad = np.ceil((height - width) / 2)
        detections[..., 0] -= pad
        detections[..., 2] -= pad

    return detections


def transform_input(img, img_size):
    data_transform = transforms.Compose([
        transforms.ToTensor()])
    height, width = img.shape[:2]
    if height > width:
        pad_top = int(np.ceil((height - width) / 2))
        pad_bot = (height - width) // 2
        img = cv2.copyMakeBorder(
            img, 0, 0, pad_top, pad_bot, cv2.BORDER_CONSTANT, value=(127.5, 127.5, 127.5))
    else:
        pad_top = int(np.ceil((width - height) / 2))
        pad_bot = (width - height) // 2
        img = cv2.copyMakeBorder(
            img, pad_top, pad_bot, 0, 0, cv2.BORDER_CONSTANT, value=(127.5, 127.5, 127.5))
    img = cv2.resize(img, (img_size, img_size))
    img = data_transform(img)
    return img


def generate_class_colors(num_classes):
    colors = np.zeros([num_classes, 3])
    for i in range(num_classes):
        colors[i] = np.random.randint(0, 266, 3)
    return colors


def get_class_names(names_path):
    with open(names_path, "r") as nf:
        names = nf.read().splitlines()
    class_to_names = {}
    for i, n in enumerate(names):
        class_to_names[i] = n
    return class_to_names


def write_fps(frame, start_time):
    x, y = 0, frame.shape[0]
    fps = round(1 / (time.time() - start_time), 2)
    stats = "FPS: {}".format(fps)
    cv2.putText(frame, stats,
                (x, y),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (255, 255, 255),
                2)
