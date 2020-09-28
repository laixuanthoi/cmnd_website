import numpy as np
import time
import cv2


class Model:
    def __init__(self, configPath, weightPath, classPath, input_size=(608, 608)):
        self.model_input_size = input_size
        self.fps = 0
        self.loadClassNames(classPath)
        self.loadModel(configPath, weightPath)

    def loadClassNames(self, classPath):
        self.class_names = open(classPath).read().strip().split("\n")

    def loadModel(self, configPath, weightPath):
        self.net = cv2.dnn.readNet(weightPath, configPath)
        self.net
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(
            size=self.model_input_size, scale=1/255.0, swapRB=True)

    def predict(self, image, confidence_threshold, nms_threshold):
        start = time.time()
        classes, scores, boxes = self.model.detect(
            image, confidence_threshold, nms_threshold)
        end = time.time()
        print("Predicted in: {}".format(end - start))
        self.fps = 1/(end - start)
        return classes, scores, boxes
