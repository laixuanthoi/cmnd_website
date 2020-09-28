import cv2
import numpy as np
# from model import Model
from upload.src.lib.model import Model
from upload.src.lib.reader import READER


line_height = 20


def swap(a, b):
    tmp = a
    a = b
    b = tmp
    return a, b


def checkoverLapse(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x11, y11, x12, y12 = x1, y1, x1 + w1, y1 + h1
    x21, y21, x22, y22 = x2, y2, x2 + w2, y2 + h2

    if (y11 > y21 and y11 < y22) or (y12 > y21 and y12 < y22):
        return True

    if (y21 > y11 and y21 < y12) or (y22 > y11 and y22 < y12):
        return True
    return False


def sortBoxPosition(arr):
    new_arr = arr
    n = len(arr)
    for i in range(0, n-1):
        for j in range(i+1, n):
            mid_i = arr[i][1] + arr[i][3]//2
            mid_j = arr[j][1] + arr[j][3]//2
            if checkoverLapse(arr[i], arr[j]):
                if arr[i][0] > arr[j][0]:
                    new_arr[i], new_arr[j] = swap(new_arr[i], new_arr[j])
            else:
                if arr[i][1] > arr[j][1]:
                    new_arr[i], new_arr[j] = swap(new_arr[i], new_arr[j])
    return new_arr


# def sort_and_overlapse(arr):
#     new_arr = []
#     n = len(arr)
#     for i in range(0, n-1):
#         for j in range(i+1, n):


# def row_overlapse(arr):
#     fpr in range(0)


class DETECTOR:
    def __init__(self, configPath, weightPath, classPath, input_size=(1024, 1024)):
        self.model = Model(configPath, weightPath,
                           classPath, input_size)
        self.crop_offset = 5

    def cropBox(self, image, boxes):
        cands = []
        for box in boxes:
            x, y, w, h = box
            x1 = x - self.crop_offset
            y1 = y - self.crop_offset
            x2 = (x + w) + self.crop_offset
            y2 = (y + h) + self.crop_offset
            cropped = image[y1:y2, x1:x2]
            cands.append(cropped)
        return cands

    def detect(self, image, confidence_threshold, nms_threshold):
        classes, scores, boxes = self.model.predict(
            image, confidence_threshold, nms_threshold)
        # self.drawing(image, classes, scores, boxes)
        hinh_candidate_boxes = []
        maso_cadidate_boxes = []
        hoten_candidate_boxes = []
        ngaysinh_candidate_boxes = []
        nguyenquan_candidate_boxes = []
        diachi_candidate_boxes = []
        H, W = image.shape[:2]
        for clss, box in zip(classes, boxes):
            if clss == 0:
                hinh_candidate_boxes.append(box)
            if clss == 1:  # maso
                maso_cadidate_boxes.append(box)
            if clss == 2:  # ho ten
                hoten_candidate_boxes.append(box)
            if clss == 3:  # ngay sinh
                ngaysinh_candidate_boxes.append(box)
            if clss == 4:  # nguyen quan dia chi
                x, y, w, h = box
                mid_y = y+h//2
                if mid_y < 215:
                    nguyenquan_candidate_boxes.append(box)
                else:
                    diachi_candidate_boxes.append(box)
                    #     if mid_y < 195:
                    #         nguyenquan_row1_candidate_boxes.append(box)
                    #     else:
                    #         nguyenquan_row2_candidate_boxes(box)
                    # else:
                    #     if mid_y < 240:
                    #         diachi_row1_candidate_boxes.append(box)
                    #     else:
                    #         diachi_row2_candidate_boxes.append(box)
        hinh_candidate_boxes = sortBoxPosition(hinh_candidate_boxes)
        maso_cadidate_boxes = sortBoxPosition(maso_cadidate_boxes)
        hoten_candidate_boxes = sortBoxPosition(hoten_candidate_boxes)
        ngaysinh_candidate_boxes = sortBoxPosition(ngaysinh_candidate_boxes)
        nguyenquan_candidate_boxes = sortBoxPosition(
            nguyenquan_candidate_boxes)
        diachi_candidate_boxes = sortBoxPosition(
            diachi_candidate_boxes)

        hinh_images = self.cropBox(image, hinh_candidate_boxes)
        maso_images = self.cropBox(image, maso_cadidate_boxes)
        hoten_images = self.cropBox(image, hoten_candidate_boxes)
        ngaysinh_images = self.cropBox(
            image, ngaysinh_candidate_boxes)
        nguyenquan_images = self.cropBox(
            image, nguyenquan_candidate_boxes)

        diachi_images = self.cropBox(
            image, diachi_candidate_boxes)

        return hinh_images, maso_images, hoten_images, ngaysinh_images, nguyenquan_images, diachi_images

    def drawing(self, image, classes, scores, boxes):
        drawed = image.copy()
        COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            # label = "%s : %f" % (self.model.class_names[classid[0]], score)
            cv2.rectangle(drawed, box, color, 1)
            # cv2.putText(drawed, label, (box[0], box[1] - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imshow("detected", drawed)
