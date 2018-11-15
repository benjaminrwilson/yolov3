import torch
import torch.nn as nn

import utils


class YoloLoss(nn.Module):

    def __init__(self, coord=.5, noobj=.5):
        super(YoloLoss, self).__init__()
        self.coord = coord
        self.noobj = noobj

    def forward(self, input, targets):
        input_corners = utils.darknet2corners(input)
        target_corners = utils.darknet2corners(targets)

        # TODO Filter object confidences
        bbox = utils.bb_iou(target_corners, input_corners[..., :4])
        max_idx = torch.argmax(bbox, 1)

        mse_loss = nn.MSELoss(reduction="sum")
        bce_loss = nn.BCELoss()
        ce_loss = nn.CrossEntropyLoss()

        assignment = input[0, max_idx, :]
        target_class = targets[..., 4].type(torch.LongTensor).unsqueeze(0)

        # TODO Fix batch dims
        targets = targets.unsqueeze(0)
        obj_targets = torch.ones(targets[..., 4].shape)
        # print("here")
        # print(assignment[..., 4].shape)
        # print(obj_targets.shape)
        # return

        coord_loss = mse_loss(assignment[..., :2], targets[..., :2])
        dim_loss = mse_loss(assignment[..., 2:4] ** .5, targets[..., 2:4] ** .5)
        obj_loss = bce_loss(assignment[..., 4], obj_targets)
        noobj_loss = 0
        class_loss = ce_loss(assignment[..., 5:], target_class)

        print(obj_loss)
        print("Class loss: {}".format(class_loss))



        # print(max_iou, max_idx)
        # print("BBOX")
        # print(bbox)
