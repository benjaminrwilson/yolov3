#!/bin/bash

if ! test -d ~/.torch/yolov3/; then mkdir ~/.torch/yolov3/; fi

wget https://pjreddie.com/media/files/yolov3.weights -O ~/.torch/yolov3/yolov3.weights
