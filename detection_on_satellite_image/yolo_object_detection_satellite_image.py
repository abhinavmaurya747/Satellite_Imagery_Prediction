import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import numpy as np
import glob
import random
from time import time,sleep
import argparse

# Load Yolo
net = cv2.dnn.readNet("data/yolov3_custom_3000.weights", "data/yolov3_custom.cfg")

# Name custom object
classes = ["MeshAntenna","Radome"]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#colors = np.random.uniform(0, 255, size=(len(classes), 3))
colors = [[255., 0., 0.],[0.,0.,255.]]

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
                # print(class_id)
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


def handle_arguments():
    global img_name
    global img_path
    # handle command line arguments
    ap = argparse.ArgumentParser()
    # adding support to import image from command line argument
    # -i IMAGE_PATH
    ap.add_argument('-i', '--image', required=True,
                    help = 'path to input image')
    args = ap.parse_args()

    # extracting the image name from the full image path
    img_path = args.image
    img_name = os.path.split(args.image)[1]


def run():
    handle_arguments()
    img = cv2.imread(img_path)
    
    preds = predict_on_img_chip(img)
    boxes = preds['boxes']
    confidences = preds['confidences']
    class_ids = preds['class_ids']
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
    for i in indexes:
        print(boxes[i[0]])
        print(confidences[i[0]], class_ids[i[0]])
    print('total detections : '+str(len(indexes)))

    font = cv2.FONT_HERSHEY_PLAIN

    # In the loop we are putting the predicted bounding boxes on the image with proper color
    # Red is for Radome and Blue is for MeshAntenna
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            print(x,y,x+w,y+h)
            
            # cv2.rectangle(image, start_point, end_point, color, thickness)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 10)
            
            # cv2.putText(image, 'label', (x,y)_coordinate, font, fontScale, color, thickness)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 5)
            
            # Annotations of the bounding boxes are being saved in a text file.
            # This text file is created in annotations/ directory with the same name as of the file.
            with open('annotations/'+img_name[:-4]+'.txt', 'a') as anot_file:
                anot_file.write('{}, {}, {}, {}, {}\n'.format(class_ids[i], x, y, w, h))



    # All the predictions that are done on the image are being saved to predictions/ directory
    # The file being created has the same name as of the original image.
    cv2.imwrite('predictions/'+img_name,img)
    print('{} of shape {} written to predictions/'.format(img_name, img.shape))

if __name__ == "__main__":
    run()