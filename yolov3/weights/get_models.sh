#!/bin/bash

if ! test -d ~/.torch/yolov3/cfg; then mkdir ~/.torch/yolov3/cfg; fi
if ! test -d ~/.torch/yolov3/weights; then mkdir ~/.torch/yolov3/weights; fi

wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O ~/.torch/yolov3/cfg/yolov3.cfg && \
wget https://pjreddie.com/media/files/yolov3.weights -O ~/.torch/yolov3/weights/yolov3.weights
