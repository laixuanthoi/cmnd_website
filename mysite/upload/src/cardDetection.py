import numpy as np
import argparse
import time
import cv2
from upload.src.lib.cropper import CROPPER
from upload.src.lib.detector import DETECTOR
import glob
import os
from upload.src.lib.reader import readText


cropper_config = {
    "classPath": "upload/src/bin/cropper/classes.names",
    "weightPath": "upload/src/bin/cropper/tiny_yolo4_darknet_backbone_2700.weights",
    "configPath": "upload/src/bin/cropper/tiny_yolo4_darknet_backbone.cfg",
    "confidence_threshold": 0.5,
    "nms_threshold": 0.5
}

detector_config = {
    "classPath": "upload/src/bin/detector/classes.names",
    "weightPath": "upload/src/bin/detector/tiny_yolo4_darknet_backbone_6500.weights",
    "configPath": "upload/src/bin/detector/tiny_yolo4_darknet_backbone.cfg",
    "confidence_threshold": 0.5,
    "nms_threshold": .5
}


cropper = CROPPER(cropper_config["configPath"],
                  cropper_config["weightPath"], cropper_config["classPath"])

detector = DETECTOR(detector_config["configPath"],
                    detector_config["weightPath"], detector_config["classPath"])


def extractInfoFromImage(path):
    image = cv2.imread(path)
    cropped_img = cropper.detectCardInImage(
        image, cropper_config["confidence_threshold"], cropper_config["nms_threshold"])
    data = {
        "info": {
            "maso": "N/A",
            "hoten": "N/A",
            "ngaysinh": "N/A",
            "nguyenquan": "N/A",
            "diachi": "N/A",
        }
    }
    if cropped_img is None:
        return data
    hinh_imgs, maso_imgs, hoten_imgs, ngaysinh_imgs, nguyenquan_imgs, diachi_imgs = detector.detect(
        cropped_img, detector_config["confidence_threshold"], detector_config["nms_threshold"])

    data["info"]["maso"] = readText(maso_imgs)
    data["info"]["hoten"] = ' '.join(readText(hoten_imgs))
    data["info"]["ngaysinh"] = '-'.join(readText(ngaysinh_imgs))
    data["info"]["nguyenquan"] = ' '.join(
        readText(nguyenquan_imgs))
    data["info"]["diachi"] = ' '.join(readText(diachi_imgs))
    return data
