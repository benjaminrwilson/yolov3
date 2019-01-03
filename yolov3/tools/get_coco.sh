#!/bin/bash

if ! [ test -d ~/datasets ]; then
    mkdir -p ~/datasets/coco
else
    echo "'datasets' directory exists, please remove or edit the bash script!"
    exit
fi

wget -c https://pjreddie.com/media/files/val2014.zip -P ~/datasets/coco
wget -c https://pjreddie.com/media/files/instances_train-val2014.zip -P ~/datasets/coco
