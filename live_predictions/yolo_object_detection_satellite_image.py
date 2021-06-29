import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import numpy as np
import glob
import random
from time import time,sleep

# Load Yolo
net = cv2.dnn.readNet("data/yolov3_custom_3000.weights", "data/yolov3_custom.cfg")

# Name custom object
classes = ["MeshAntenna","Radome"]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#colors = np.random.uniform(0, 255, size=(len(classes), 3))


def predict_on_img_chip(img):
    # Detecting objects
    h,w,c = img.shape       #h,w,c are the original height width and no. of channels of the original image. By doing this we get back our original image
    img = cv2.resize(img, (w,h), fx=0.4, fy=0.4)

    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    #indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    #print(indexes)
    #print(boxes)
    #print(confidences)

    prediciton_values = {'boxes':boxes, 'confidences':confidences, 'class_ids':class_ids}
    return prediciton_values


img = cv2.imread('sliced_.jpg')

#predict_on_img_chip(img)