#!/bin/bash

if ! [ test -d ~/coco ]; then
    mkdir -p ~/coco
else
    echo "'coco' directory exists, please remove or edit the bash script!"
    exit
fi

wget -c http://images.cocodataset.org/zips/val2014.zip -P ~/coco
wget -c https://pjreddie.com/media/files/instances_train-val2014.zip -P ~/coco
