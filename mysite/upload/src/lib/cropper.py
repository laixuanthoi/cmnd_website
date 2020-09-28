import cv2
import numpy as np
# from model import Model
from upload.src.lib.model import Model
from math import sqrt


def cal_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return sqrt((x1-x2)**2 + (y1-y2)**2)


def findClosestPoint(arr, p):
    if len(arr) <= 0:
        return None
    tmp_dis = cal_distance(arr[0], p)
    tmp_p = arr[0]
    for i in range(1, len(arr)):
        dis = cal_distance(arr[i], p)
        if dis < tmp_dis:
            tmp_p = p
            tmp_dis = dis
    return tmp_p


class CROPPER:
    def __init__(self, configPath, weightPath, classPath):
        self.cardWidth = 430
        self.cardHeight = 270  # 642
        self.model = Model(configPath, weightPath, classPath)
        self.count_cropped = 1

    def warpPerspective(self, image, points):
        # top left - top right - bot right - bot left
        src_points = np.float32(points)
        dst_points = np.float32([[0, 0],                 [self.cardWidth, 0],
                                 [0, self.cardHeight],   [self.cardWidth, self.cardHeight]])
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(image, matrix, (self.cardWidth, self.cardHeight))

    def detectCardInImage(self, image, confidence_threshold, nms_threshold):
        classes, scores, boxes = self.model.predict(
            image, confidence_threshold, nms_threshold)
        # self.drawing(image, classes, scores, boxes)
        off_set = 5
        tls = []
        trs = []
        bls = []
        brs = []

        for clss, box in zip(classes, boxes):
            c_x = (box[0] + (box[0]+box[2]))//2
            c_y = (box[1] + (box[1]+box[3]))//2
            if clss[0] == 0:
                tls.append((c_x - off_set, c_y - off_set))
            if clss[0] == 1:
                trs.append((c_x + off_set, c_y - off_set))
            if clss[0] == 2:
                bls.append((c_x - off_set, c_y + off_set))
            if clss[0] == 3:
                brs.append((c_x + off_set, c_y + off_set))

        if len(tls) <= 0 or len(trs) <= 0 or len(bls) <= 0 or len(brs) <= 0:
            return None
        four_point_corners = [
            tls[0], trs[0], bls[0], brs[0]
        ]
        cropped = self.warpPerspective(image, four_point_corners)
        self.count_cropped += 1
        return cropped

    def drawing(self, image, classes, scores, boxes):
        drawed = image.copy()
        COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (self.model.class_names[classid[0]], score)
            cv2.rectangle(drawed, box, color, 1)
            cv2.putText(drawed, label, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imshow("drawed image", drawed)
