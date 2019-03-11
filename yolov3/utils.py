import os
import time

import cv2
import numpy as np
import PIL
import torch
from torchvision.transforms import Pad, Resize, ToTensor

from localization.bboxes import BBoxes, CoordType


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


def center2xyxy(coords):
    bottom_right = coords[..., :2].clone() + (coords[..., 2:4] - 1) / 2
    coords[..., :2] -= coords[..., 2:4] / 2
    coords[..., 2:4] = bottom_right
    return coords


def bb_iou(bb1, bb2):
    bb1_x1, bb1_y1, bb1_x2, bb1_y2 = bb1[..., :4].split(1, dim=-1)
    bb2_x1, bb2_y1, bb2_x2, bb2_y2 = bb2[..., :4].split(1, dim=-1)

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


def nms(x, obj_thresh, nms_thresh):
    n_batches = x.shape[0]
    batches = []
    for i in range(n_batches):
        preds = x[i, x[i, ..., 4] > obj_thresh]
        dets = []
        if preds.shape[0] > 0:
            preds = center2xyxy(preds)
            class_conf, pred_class = torch.max(preds[..., 5:], 1)

            class_conf = class_conf.float().unsqueeze(1)
            pred_class = pred_class.float().unsqueeze(1)

            preds = torch.cat((preds[..., :5], class_conf, pred_class), 1)
            img_classes = torch.unique(preds[..., -1])

            for img_class in img_classes:
                pred_cls = preds[preds[..., -1] == img_class]
                conf_idx = torch.sort(pred_cls[..., 4],
                                      descending=True)[1]
                pred_cls = pred_cls[conf_idx]

                dets = _nms_helper(dets, pred_cls, nms_thresh)
            batches.append(torch.stack(dets))
    return batches


def _nms_helper(detections, pred_cls, nms_thresh):
    while pred_cls.shape[0] > 0:
        detections.append(pred_cls[0])
        if pred_cls.shape[0] == 1:
            break
        ious = bb_iou(pred_cls[0], pred_cls[1:])
        iou_mask = (ious < nms_thresh).squeeze(1)
        pred_cls = pred_cls[1:][iou_mask]
    return detections


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


def draw_bboxes(bboxes,
                img,
                class_colors,
                color=(255, 255, 255),
                line_type=cv2.LINE_AA,
                thickness=3):
    coords = bboxes.coords
    labels = bboxes.attrs["labels"].long()
    for (x1, y1, x2, y2), label in zip(coords, labels):
        cv2.rectangle(img,
                      (x1, y1),
                      (x2, y2),
                      class_colors[label.item()],
                      lineType=line_type,
                      thickness=thickness)


def _write_detection(img,
                     detections,
                     class_colors,
                     class_to_names,
                     line_type=cv2.LINE_AA):
    for d in detections:
        x1, y1, x2, y2 = d[2:].cpu().numpy()
        class_pred = int(d[0].cpu().numpy())

        height, width = img.shape[0:2]
        scale = max((x2 - x1) / width, (y2 - y1) / height)
        conf = str(int(100 * d[1].cpu().numpy())) + "%"
        img = cv2.rectangle(img,
                            (x1, y1),
                            (x2, y2),
                            class_colors[class_pred],
                            lineType=line_type,
                            thickness=10 * d[1])
        img = cv2.putText(img,
                          class_to_names[class_pred] + "|" + conf,
                          (x1, y1 - 20 * d[1]),
                          cv2.FONT_HERSHEY_DUPLEX,
                          scale,
                          (255, 255, 255),
                          1,
                          lineType=line_type)
    return img


def transform_detections(batches, w, h, dw, dh, size):
    res = []
    for i, bboxes in enumerate(batches):
        bboxes[..., 0] -= dw[i] // 2
        bboxes[..., 1] -= dh[i] // 2
        bboxes[..., 2] -= dw[i] // 2
        bboxes[..., 3] -= dh[i] // 2
        ratio = max(w[i], h[i]) / size

        bboxes[..., :4] *= ratio
        coords = bboxes[..., :4]
        confidences = bboxes[..., 4] * bboxes[..., 5]
        labels = bboxes[..., 6]
        bboxes = BBoxes(bboxes[..., :4],
                        CoordType.XYXY, (w, h))
        bboxes.attrs["confidences"] = confidences
        bboxes.attrs["labels"] = labels
        bboxes.coords = coords
        res.append(bboxes)
    return res


def transform_input(img, size):
    if isinstance(img, np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(img)
    w, h = img.size
    target = int((size / max(w, h)) * min(w, h))
    img = Resize(target)(img)

    dw = size - img.size[0]
    dh = size - img.size[1]
    padding = (dw // 2, dh // 2, dw - dw // 2, dh - dh // 2)
    img = Pad(padding)(img)
    return ToTensor()(img), w, h, dw, dh


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
    cv2.putText(frame,
                stats,
                (x, y),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (255, 255, 255),
                2)
