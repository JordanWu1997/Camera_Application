#!/usr/bin/env bash

###########################################################
# Author      : Kuan-Hsien Wu
# Contact     : jordankhwu@gmail.com
# Datetime    : 2024-12-28 20:40:10
# Description : Download model from YOLOv11
#               https://github.com/ultralytics/ultralytics?tab=readme-ov-file
###########################################################

mkdir -p ./weights

wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -O ./weights/yolo11n.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt -O ./weights/yolo11n-seg.pt
#wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt -O ./weights/yolo11n-cls.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt -O ./weights/yolo11n-pose.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.pt -O ./weights/yolo11n-obb.pt
