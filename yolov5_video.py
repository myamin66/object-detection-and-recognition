#Download droidcam
#Get IP-Port of the streamed cam

# import the opencv library
#pip install opencv-python==3.7.0
import cv2
import torch

torch.cuda.is_available()
import numpy as np
import math
import cv2 as cv
from cvzone import HandTrackingModule
import keras
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom
detector = HandTrackingModule.HandDetector()
# define a video capture object
vid = cv2.VideoCapture(0)

while(vid.isOpened()):

    # Capture the video frame

    # by frame
    ret, frame = vid.read()
    results = model(frame)
    ds = results.pandas().xyxy[0]
    print("Detected objects by Yolo", ds)

    for i in range(ds.shape[0]):
        hands,frame =  detector.findHands(frame)
        x,y,x2,y2 = ds.iloc[i].xmin, ds.iloc[i].ymin, ds.iloc[i].xmax, ds.iloc[i].ymax
        x=int(x)
        y=int(y)
        x2=int(x2)
        y2=int(y2)
        cv2.rectangle(frame, (x,y), (x2,y2),(0,0,255),1)
        

    # Display the resulting frame
  
    cv2.imshow('frame', frame)
    if(hands):
        obj = hands.__getitem__(0)
        boxValue = obj.get('bbox')
        print("Bounding box of hands" ,boxValue)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
