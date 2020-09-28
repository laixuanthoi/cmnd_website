import numpy as np
import argparse
import time
import cv2
# from lib.cropper import CROPPER
# from lib.detector import DETECTOR
# from lib.reader import READER
from upload.src.lib.cropper import CROPPER
from upload.src.lib.detector import DETECTOR
from upload.src.lib.reader import READER
import glob
import os


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

reader = READER()


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

    data["info"]["maso"] = reader.readMultiNumber(maso_imgs)
    data["info"]["hoten"] = ' '.join(reader.readMultiText(hoten_imgs))
    data["info"]["ngaysinh"] = '-'.join(reader.readMultiNumber(ngaysinh_imgs))
    data["info"]["nguyenquan"] = ' '.join(
        reader.readMultiText(nguyenquan_imgs))
    data["info"]["diachi"] = ' '.join(reader.readMultiText(diachi_imgs))
    return data
