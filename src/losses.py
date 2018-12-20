import cv2
import torch
from torch import nn

import utils
import sys


class YoloLoss(nn.Module):

    def __init__(self, coord=.5, noobj=.5):
        super(YoloLoss, self).__init__()
        self.coord = coord
        self.noobj = noobj

    def forward(self, x, targets, img):
        print("Forward Loss")
        input_corners = utils.darknet2corners(x)
        target_corners = utils.darknet2corners(targets)

        bbox = utils.bb_iou(target_corners, input_corners[..., :4])

        obj_conf = bbox.max().unsqueeze(0)
        obj_mask = (bbox == obj_conf).type(torch.ByteTensor)

        noobj_mask = (obj_mask ^ 1)

        sys.exit(0)

        # Find object with max IoU
        obj_assignment = x[obj_mask]
        # Complement of max IoU mask
        noobj_assignment = x[noobj_mask]

        # Make losses
        mse_loss = nn.MSELoss(reduction="sum")
        bce_loss = nn.BCELoss()
        ce_loss = nn.CrossEntropyLoss()

        # TODO Fix batch dims
        targets = targets.unsqueeze(0)

        # Term 1
        coord_loss = mse_loss(obj_assignment[..., :2], targets[..., :2])
        # Term 2
        dim_loss = mse_loss(obj_assignment[..., 2:4] ** .5,
                            targets[..., 2:4] ** .5)
        # Term 3
        obj_loss = bce_loss(obj_assignment[..., 4], obj_conf)
        # Term 4
        noobj_loss = self.noobj * \
            bce_loss(noobj_assignment[..., 4], bbox[noobj_mask])

        # Term 5
        cls_loss = ce_loss(
            obj_assignment[..., 5:], targets[..., 4].type(torch.LongTensor))

        loss = coord_loss + dim_loss + obj_loss + noobj_loss + cls_loss

        losses = "Coord loss: {} | Dim loss: {} | Obj loss: {} | Noobj loss: {} | Class loss: {} | Total loss: {}".format(
            coord_loss, dim_loss, obj_loss, noobj_loss, cls_loss, loss)
        print(losses)

        targets = targets.squeeze(0)
        x = utils.transform_detections(
            x, 1280, 720, 416, is_corners=True)
        targets = utils.transform_detections(
            targets, 1280, 720, 416, is_corners=True)

        x1, y1, x2, y2 = target_corners[..., :4]
        x11, y11, x22, y22 = input_corners[obj_mask][0][..., :4]

        img = cv2.rectangle(img, (x11, y11), (x22, y22), (0, 0, 0))
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255))

        cv2.imshow("Image", img)
        cv2.waitKey(0)
