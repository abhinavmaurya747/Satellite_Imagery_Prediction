import os
import re
# Setting the OPENCV_IO_MAX_IMAGE_PIXELS to 2^40 so that large images can also be imported 
# It does not works on linux
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import numpy as np
import glob
import random
from time import time,sleep
from yolo_object_detection_satellite_image import predict_on_img_chip
import argparse


# handle command line arguments
ap = argparse.ArgumentParser()
# adding support to import image from command line argument
# -i IMAGE_PATH
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
args = ap.parse_args()


# extracting the image name from the full image path
img_name = os.path.split(args.image)[1]#re.split(r"\\", args.image)[-1]
img_path = 'satellite_images'
classes = ["MeshAntenna","Radome"]

# Uncomment if you are want random colors
#colors = np.random.uniform(0, 255, size=(len(classes), 3))
colors = [[255., 0., 0.],[0., 0., 255.]]

slicing_size = 1664 #1664 #1664 #1024 #2080 #416 #1664
start_h, start_w = 0, 0

t0=time()
#img = cv2.imread(os.path.join(img_path,img_name))
img = cv2.imread(args.image)
orig_img = img.copy()
print('Image is imported.')
width,height,channels = img.shape

print(img.shape)

if img.shape[0] <=slicing_size and img.shape[0] >= 416:
    slicing_size = slicing_size//2
elif img.shape[0] >= slicing_size:
	img = cv2.resize(img, (slicing_size*(height//slicing_size),slicing_size*(width//slicing_size)), fx=0.4, fy=0.4)
	w_new, h_new, c_new = img.shape
	print(img.shape)


t0 = time()

tiles_row = height//slicing_size
tiles_col = width//slicing_size

out = list(np.zeros((width//slicing_size,height//slicing_size)))
out = [list(i) for i in out]

boxes = []
confidences = []
class_ids = []
tiles = []

init_time = time()
last_time = time()

# Breaking image into chips of size = slicing_size and then doing predictions on them
# All the predictions are stored in boxes, confidences, class_ids, tiles
i=0
j=0
for i in range(tiles_col-start_w):
    for j in range(tiles_row-start_h):
        #print(i,j)
        out[i][j] = img[i*slicing_size+start_w:i*slicing_size+slicing_size+start_w,j*slicing_size+start_h:j*slicing_size+slicing_size+start_h,:]
        prediction_values = predict_on_img_chip(out[i][j])

        chip_boxes = prediction_values['boxes']
        chip_confidences = prediction_values['confidences']
        chip_class_ids = prediction_values['class_ids']
        for ind in range(len(chip_boxes)):
            x, y, w, h = chip_boxes[ind]
            chip_boxes[ind] = x + j*slicing_size, y + i*slicing_size, w, h
            boxes.append(chip_boxes[ind])
            confidences.append(chip_confidences[ind])
            class_ids.append(chip_class_ids[ind])
            tiles.append((tiles_col,tiles_row))
        if (i*tiles_row+j)%64==0:
            if (i*tiles_row+j)==0:
                pass
            else:
                print("No. of chips processed : "+str(i*tiles_row+j))
        # Uncomment to find out how much time is it taking to process one chip
        # print(time()-last_time)
        # last_time = time()

"""
start_w,start_h = 416, 416
for i in range(tiles_col):
    for j in range(tiles_row):
        #print(i,j)
        out[i][j] = img[i*slicing_size+start_w:i*slicing_size+slicing_size+start_w,j*slicing_size+start_h:j*slicing_size+slicing_size+start_h,:]
        prediction_values = predict_on_img_chip(out[i][j])

        chip_boxes = prediction_values['boxes']
        chip_confidences = prediction_values['confidences']
        chip_class_ids = prediction_values['class_ids']
        for ind in range(len(chip_boxes)):
            x, y, w, h = chip_boxes[ind]
            chip_boxes[ind] = x + j*slicing_size-start_h, y + i*slicing_size-start_w, w, h
            boxes.append(chip_boxes[ind])
            confidences.append(chip_confidences[ind])
            class_ids.append(chip_class_ids[ind])
            tiles.append((tiles_col,tiles_row))
        if (i*tiles_row+j)%64==0:
            if (i*tiles_row+j)==0:
                pass
            else:
                print("No. of chips processed : "+str(i*tiles_row+j))
"""

# Dumping out results to the console
print("Processing took: ",time()-init_time)
print("Total no. of chips processed : "+str(i*tiles_row+j+1))
# NMS - NonMaxSupression is an algorithm that discards the unwanted predictions
# After NMS only the the boxes that are accurate enough with very less collisions remain
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3) #0.4, 0.3

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
        x = int(x/h_new * height)
        y = int(y/w_new * width)
        w = int(w/h_new * height)
        h = int(h/w_new * width)

        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        print(x,y,x+w,y+h)
        print(confidences[i])
        
        # cv2.rectangle(image, start_point, end_point, color, thickness)
        #cv2.rectangle(img, (x, y), (x + w, y + h), color, 10)
        cv2.rectangle(orig_img, (x, y), (x + w, y + h), color, 10)

        # cv2.putText(image, 'label', (x,y)_coordinate, font, fontScale, color, thickness)
        #cv2.putText(img, label, (x, y + 30), font, 3, color, 5)
        cv2.putText(orig_img, label, (x, y + 30), font, 3, color, 5)
		

		# Annotations of the bounding boxes are being saved in a text file.
		# This text file is created in annotations/ directory with the same name as of the file.
        with open('annotations/'+img_name[:-4]+'.txt', 'a') as anot_file:
            anot_file.write('{}, {}, {}, {}, {}\n'.format(class_ids[i], x, y, w, h))



# All the predictions that are done on the image are being saved to predictions/ directory
# The file being created has the same name as of the original image.
#cv2.imwrite('predictions/'+img_name,img)
#print('{} of shape {} written to predictions/'.format(img_name, img.shape))
cv2.imwrite('predictions/'+img_name,orig_img)
print('{} of shape {} written to predictions/'.format(img_name, orig_img.shape))

#print(time()-t0)


















