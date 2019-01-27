import os
import time

import cv2
import numpy as np
import PIL
import torch
from torchvision import transforms

from yolov3.src.get_image_size import get_image_size


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


def bb_iou(bb1, bb2):
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
    _write_detection(img, detections, class_colors, class_to_names)

    img_name = img_path.split("/")[-1]
    dst = os.path.join(dst, img_name)
    cv2.imwrite(dst, img)


def write_detections_cam(detections, img, size, class_colors, class_to_names):
    if len(detections) > 0:
        height, width = img.shape[0:2]
        _write_detection(img, detections, class_colors, class_to_names)


def show_bboxes(bboxes, img):
    for x1, y1, x2, y2 in bboxes.coords:
        img = cv2.rectangle(img, (x1, y1), (x2, y2),
                            (255, 255, 255),
                            lineType=cv2.LINE_AA,
                            thickness=3)
    return img


def _write_detection(img, detections, class_colors, class_to_names):
    for d in detections:
        x1, y1, x2, y2 = d[2:].cpu().numpy()
        class_pred = int(d[0].cpu().numpy())

        height, width = img.shape[0:2]
        scale = max((x2 - x1) / width, (y2 - y1) / height)
        conf = str(int(100 * d[1].cpu().numpy())) + "%"
        img = cv2.rectangle(img, (x1, y1), (x2, y2),
                            class_colors[class_pred],
                            lineType=cv2.LINE_AA,
                            thickness=10 * d[1])
        img = cv2.putText(img, class_to_names[class_pred] + "|" + conf,
                          (x1, y1 - 20 * d[1]),
                          cv2.FONT_HERSHEY_DUPLEX,
                          scale,
                          (255, 255, 255),
                          1,
                          lineType=cv2.LINE_AA)
    return img


def transform_detections(detections, width, height, size, is_corners=True):
    ratio = max(width, height) / size
    detections[..., :4] *= ratio
    pad = np.abs(height - width) / 2
    if width > height:
        detections[..., 1] -= pad
        if is_corners:
            detections[..., 3] -= pad
    else:
        detections[..., 0] -= pad
        if is_corners:
            detections[..., 2] -= pad
    return detections


def transform_darknet2padded2norm(detections, width, height, size):
    ratio = max(width, height) / size
    detections[..., :4] *= ratio
    if width > height:
        pad = np.ceil((width - height) / 2)
        detections[..., 1] -= pad
    else:
        pad = np.ceil((height - width) / 2)
        detections[..., 0] -= pad
    return detections


def transform_norm2padded(detections, width, height, size):
    detections[..., 0] *= width
    detections[..., 1] *= height
    detections[..., 2] *= width
    detections[..., 3] *= height
    if width > height:
        pad = np.ceil((width - height) / 2)
        detections[..., 1] += pad
    else:
        pad = np.ceil((height - width) / 2)
        detections[..., 0] += pad
    ratio = size / max(width, height)
    detections[..., :4] *= ratio
    return detections


def transform_input(img, size, color=(127.5, 127.5, 127.5)):
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ratio = size / max(img.shape[:2])
    img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)

    dw = size - img.shape[1]
    dh = size - img.shape[0]
    padding = [dh // 2, dh - dh // 2, dw // 2, dw - dw // 2]
    img = cv2.copyMakeBorder(img, *padding, cv2.BORDER_CONSTANT, color)
    return transforms.ToTensor()(img)


def generate_class_colors(num_classes):
    colors = np.zeros([num_classes, 3])
    for i in range(num_classes):
        colors[i] = np.random.randint(0, 256, 3)
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


def calculate_map(ann_path, img_path, results, device, class_colors, class_to_names, size, dst):
    for img_name, detections in results.items():
        ann_name = img_name.split("/")[-1].replace(".jpg", ".txt")
        abs_ann_path = os.path.join(ann_path, ann_name)

        info = "Processing Image {}".format(img_name)
        print(info)

        ground_truths = []
        with open(abs_ann_path, "r") as ann_file:
            lines = ann_file.read().splitlines()
            abs_img_name = os.path.join(img_path, img_name)
            img_width, img_height = get_image_size(abs_img_name)

            # Iterate over annotations
            for line in lines:
                class_label, x, y, width, height = [
                    float(x) for x in line.split(" ")]

                # Convert coordinates to top left and bottom right corners
                x1, y1, x2, y2 = darknet2abs_corners(float(x), float(y), float(
                    width), float(height), img_width, img_height)
                coords = torch.Tensor([x1, y1, x2, y2], device=device)

                tp, fp, fn = 0, 0, 0
                # Get the detections that match the class label
                cls_index = detections[..., -1] == class_label
                detection_cls = detections[cls_index]
                if detection_cls.shape[0] == 0:
                    continue

                # Sort the detections by class confidence
                conf_idx = torch.sort(
                    detections[cls_index][..., 5],
                    descending=True)[1]
                detection_cls = detection_cls[conf_idx]

                # Get ious with respect to ground truth label
                ious = bb_iou(coords, detection_cls)
                detections = detections[ious > .5, ...][1:, ...]
                ground_truths.append([x1, y1, x2, y2, class_label])
        ground_truths = torch.Tensor(ground_truths)


def darknet2abs_corners(x, y, width, height, img_width, img_height):
    width *= img_width
    height *= img_height
    x *= img_width
    y *= img_height

    x1 = x - (width / 2)
    y1 = y - (height / 2)
    x2 = x + (width / 2)
    y2 = y + (height / 2)
    return x1, y1, x2, y2


def get_targets(opts, ann_name, img_name):
    ann_path = opts.ann_path
    abs_ann_path = os.path.join(ann_path, ann_name)
    w, h = get_image_size(img_name)
    with open(abs_ann_path, "r") as ann_file:
        lines = ann_file.read().splitlines()

        res = []
        for line in lines:
            det = line.split(" ")
            det = [float(x) for x in det]
            det = det[1:] + [det[0]]
            res.append(det)
        res = torch.Tensor(res)
    res = transform_norm2padded(res, w, h, opts.size)
    return res
