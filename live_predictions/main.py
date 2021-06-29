import numpy as np
import pyscreenshot as ImageGrab
import cv2
import time
import tkinter
from yolo_object_detection_satellite_image import predict_on_img_chip
import sys

# making the program DPI aware on Windows. Doing this will make sure that tkinter gets the windows resolution
# Uncomment if using on windows
#"""
from ctypes import windll
user32 = windll.user32
user32.SetProcessDPIAware()
#"""

font = cv2.FONT_HERSHEY_PLAIN

#colors = np.random.uniform(0, 255, size=(len(classes), 3))
colors = np.array([[255.,0.,0.],[0.,0.,255.]])
        
cv2.useOptimized()

def screen_record(): 
    """
        This function starts recording the left half of the screen and then feed it to the YOLOV3 network 
        and the network starts predicting. The results are shown live on the right window named "Live Testing"
    """

    # start_w, start_h are the coordinates from where we have to start capturing the screen. (0,0) is the leftmost point on your screen.
    start_w,start_h = 0,0
    # get_screen_resolution() returns the exact screen resulotion of your system
    width,height = get_screen_resolution()
    winname = "Live Testing"
    #cv2.namedWindow(winname)	# Create a named window
    #cv2.moveWindow(winname, width//2,0)	# Move cv window to right side
    #cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    last_time = time.time()

    while(True):
        #capturing the left half of the display
        frame_from_screen =  np.array(ImageGrab.grab(bbox=(start_w, start_h, width//2-start_w, height-start_h)))
        print(frame_from_screen.shape)
        print('loop took {} seconds'.format(time.time()-last_time))

        
        # Processing Detections
        prediction_values = predict_on_img_chip(cv2.cvtColor(frame_from_screen, cv2.COLOR_BGR2RGB))
        boxes = prediction_values['boxes']
        confidences = prediction_values['confidences']
        class_ids = prediction_values['class_ids']
        classes = ["MeshAntenna","Radome"]

        # Non Max supressions on the detections to remove noisy and overlapping detections
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(frame_from_screen, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame_from_screen, label, (x, y + 30), font, 3, color, 2)

        #sys.stdout.flush()
        #GUI_support.update_frame(cv2.cvtColor(frame_from_screen, cv2.COLOR_BGR2RGB))
        #cv2.useOptimized()
        
        # Showing the processed detections in a opencv window
        cv2.imshow(winname, cv2.cvtColor(frame_from_screen, cv2.COLOR_BGR2RGB))

        last_time = time.time()
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        

def get_screen_resolution():
	root = tkinter.Tk()
	width = root.winfo_screenwidth()
	height = root.winfo_screenheight()
	return width,height

screen_record()

