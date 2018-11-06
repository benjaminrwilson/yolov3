import cv2
import numpy as np

import torch
import torch.nn as nn

import utils


class Shortcut(nn.Module):
    def __init__(self):
        super(Shortcut, self).__init__()

    def forward(self, x):
        return x


class Route(nn.Module):
    def __init__(self):
        super(Route, self).__init__()

    def forward(self, x):
        return x


class Yolo(nn.Module):
    def __init__(self, anchors, num_classes, input_width, input_height, device):
        super(Yolo, self).__init__()
        self.anchors = anchors
        self.device = device
        self.num_classes = num_classes
        self.input_width = input_width
        self.input_height = input_height

    def forward(self, x):
        batch_size, grid_size = x.shape[0], x.shape[2]
        grid_attr = self.num_classes + 5
        stride = self.input_width // grid_size
        n_anchors = len(self.anchors)

        x = x.view(batch_size, n_anchors, grid_attr, grid_size,
                   grid_size).permute(0, 1, 3, 4, 2).contiguous()

        anchors = torch.FloatTensor(
            [(a[0] / stride, a[1] / stride) for a in self.anchors]).to(self.device)

        grid_offsets_x = torch.arange(grid_size).repeat(grid_size, 1).view(
            1, 1, grid_size, grid_size).float().to(self.device)
        grid_offsets_y = torch.arange(grid_size).repeat(grid_size, 1).t().view(
            1, 1, grid_size, grid_size).float().to(self.device)

        x[..., 0] = torch.sigmoid(x[..., 0]) + grid_offsets_x
        x[..., 1] = torch.sigmoid(x[..., 1]) + grid_offsets_y
        x[..., 2] = torch.exp(x[..., 2]) * \
            anchors[:, 0].view(1, n_anchors, 1, 1)
        x[..., 3] = torch.exp(x[..., 3]) * \
            anchors[:, 1].view(1, n_anchors, 1, 1)
        x[..., 4] = torch.sigmoid(x[..., 4])
        x[..., 5:] = torch.sigmoid(x[..., 5:])

        x[..., :4] *= stride
        x = torch.cat((x[..., :4].view(batch_size, -1, 4),
                       x[..., 4].view(batch_size, -1, 1),
                       x[..., 5:].view(batch_size, -1, self.num_classes)), -1)
        return x


class Darknet(nn.Module):
    def __init__(self, cfg_file, weights_file, nms_thresh, obj_thresh, size, device):
        super(Darknet, self).__init__()
        self.blocks = utils.parse_cfg(cfg_file)
        self.net_meta, self.layers, self.num_classes = create_layers(
            self.blocks, size, device)
        self.nms_thresh = nms_thresh
        self.obj_thresh = obj_thresh

        self.load_weights(weights_file)

    def forward(self, x):
        blocks = self.blocks[1:]
        outputs = []

        detections = []
        for i, block in enumerate(blocks):
            l_type = block["type"]
            if l_type == "convolutional" or l_type == "upsample":
                x = self.layers[i](x)
            elif l_type == "route":
                layer_i = [int(x) for x in block['layers'].split(',')]
                x = torch.cat([outputs[i] for i in layer_i], 1)
            elif l_type == "shortcut":
                idx = int(block["from"])
                x = outputs[-1] + outputs[idx]
            elif l_type == "yolo":
                x = self.layers[i](x)
                detections.append(x)
            outputs.append(x)

        x = torch.cat(detections, 1)
        return self.nms(x)

    def load_weights(self, weights_file):
        with open(weights_file, "rb") as wf:
            np.fromfile(wf, np.int32, 5)
            weights = np.fromfile(wf, np.float32)

            ptr = 0
            n_layers = len(self.layers)
            for i in range(n_layers):
                b_type = self.blocks[i + 1]["type"]
                if b_type == "convolutional":
                    batch_normalize = "batch_normalize" in self.blocks[i + 1]
                    layer = self.layers[i]

                    convolutional = layer[0]
                    if batch_normalize:
                        batch_norm = layer[1]
                        n_biases = batch_norm.bias.numel()
                        tensor_shape = batch_norm.bias.data.shape
                        biases = torch.as_tensor(
                            weights[ptr: ptr + n_biases]).view(tensor_shape)
                        ptr += n_biases

                        tensor_shape = batch_norm.weight.data.shape
                        layer_weights = torch.as_tensor(
                            weights[ptr: ptr + n_biases]).view(tensor_shape)
                        ptr += n_biases

                        tensor_shape = batch_norm.running_mean.shape
                        running_mean = torch.as_tensor(
                            weights[ptr: ptr + n_biases]).view(tensor_shape)
                        ptr += n_biases

                        tensor_shape = batch_norm.running_var.shape
                        running_var = torch.as_tensor(
                            weights[ptr: ptr + n_biases]).view(tensor_shape)
                        ptr += n_biases

                        batch_norm.bias.data.copy_(biases)
                        batch_norm.weight.data.copy_(layer_weights)
                        batch_norm.running_mean.copy_(running_mean)
                        batch_norm.running_var.copy_(running_var)
                    else:
                        n_biases = convolutional.bias.numel()
                        tensor_shape = convolutional.bias.shape
                        biases = torch.as_tensor(
                            weights[ptr: ptr + n_biases]).view(tensor_shape)
                        convolutional.bias.data.copy_(biases)
                        ptr += n_biases
                    n_weights = convolutional.weight.numel()
                    tensor_shape = convolutional.weight.shape
                    layer_weights = torch.as_tensor(
                        weights[ptr: ptr + n_weights]).view(tensor_shape)
                    convolutional.weight.data.copy_(layer_weights)
                    ptr += n_weights

    def nms(self, x, multi_nms=False):
        n_batches = x.shape[0]
        for _ in range(n_batches):
            preds = x[x[..., 4] > self.obj_thresh]
            detections = []
            if preds.shape[0] > 0:
                preds = utils.darknet2corners(preds)
                max_conf_cls, max_conf = torch.torch.max(preds[..., 5:], 1)

                max_conf_cls = max_conf_cls.float().unsqueeze(1)
                max_conf = max_conf.float().unsqueeze(1)

                preds = torch.cat((preds[..., :5], max_conf_cls, max_conf), 1)
                img_classes = torch.unique(preds[..., -1])

                if multi_nms:
                    for img_cls in img_classes:
                        pred_cls = preds[preds[..., -1] == img_cls]
                        conf_idx = torch.sort(
                            preds[preds[..., -1] == img_cls][..., 4],
                            descending=True)[1]
                        pred_cls = pred_cls[conf_idx]
                else:
                    conf_idx = torch.sort(
                        preds[..., 4], descending=True)[1]
                    pred_cls = preds[conf_idx]

                detections = self._nms_helper(
                    detections, pred_cls, self.nms_thresh)
                detections = torch.cat(detections, 0)
        return detections

    def _nms_helper(self, detections, pred_cls, nms_thresh):
        while pred_cls.shape[0] > 0:
            detections.append(pred_cls[0])
            if pred_cls.shape[0] == 1:
                break
            ious = utils.bb_nms(pred_cls[0], pred_cls[1:])
            iou_mask = ious < nms_thresh
            pred_cls = pred_cls[1:][iou_mask]
        return detections


def create_layers(cfg, size, device):
    net_meta = cfg[0]
    modules = nn.ModuleList()

    in_channels = [3]
    for i, layer in enumerate(cfg[1:]):
        module = nn.Sequential()
        layer_type = layer["type"]
        out_channels = None
        if layer_type == "convolutional":
            out_channels = int(layer["filters"])
            kernel_size = int(layer["size"])
            padding = int(layer["pad"])
            stride = int(layer["stride"])
            padding = (kernel_size - 1) // 2 if padding else 0

            bn = "batch_normalize" in layer

            name = "conv2d_{}".format(i)
            module.add_module(name, nn.Conv2d(
                in_channels[-1],
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=not bn))

            if bn:
                name = "batch_norm_{}".format(i)
                module.add_module(name, nn.BatchNorm2d(out_channels))

            if layer["activation"] == "leaky":
                name = "leaky_relu_{}".format(i)
                module.add_module(name, nn.LeakyReLU(0.1, inplace=True))

        elif layer_type == "upsample":
            name = "upsample_{}".format(i)
            module.add_module(name, nn.Upsample(
                mode="nearest", scale_factor=2))
        elif layer_type == "route":
            indices = layer["layers"].split(",")
            indices = [int(idx.strip()) for idx in indices]

            for j, idx in enumerate(indices):
                if idx < 0:
                    indices[j] = idx
                else:
                    indices[j] = (idx - i)

            out_channels = sum([in_channels[idx] for idx in indices])
            name = "route_{}".format(i)
            module.add_module(name, Route())
        elif layer_type == "shortcut":
            name = "shortcut_{}".format(i)
            module.add_module(name, Shortcut())
        elif layer_type == "yolo":
            masks = [int(mask) for mask in layer["mask"].split(",")]
            anchors = layer["anchors"].split(",")
            anchors = [int(anch.strip()) for anch in anchors]
            anchors = [(anchors[2 * masks[i]], anchors[2 * masks[i] + 1])
                       for i in range(len(masks))]

            input_width = size
            input_height = size
            num_classes = int(layer["classes"])

            name = "yolo_{}".format(i)
            module.add_module(name, Yolo(
                anchors, num_classes, input_width, input_height, device))
        if out_channels:
            in_channels.append(out_channels)
        else:
            in_channels.append(in_channels[-1])
        modules.append(module)
    return net_meta, modules, num_classes
