import pyrealsense2 as rs
import cv2
from pathlib import Path
import numpy as np
from win32com.client import Dispatch
import time

# constants
base_path = Path(__file__).parent
weights_path = "D:\BSCS 7\Machine Learning\Project\Real-time-Object-Detection-Flask-OpenCV-YoloV3-main\Real-time-Object-Detection-Flask-OpenCV-YoloV3-main\models\yolov3.weights"
cfg_path = "D:\BSCS 7\Machine Learning\Project\Real-time-Object-Detection-Flask-OpenCV-YoloV3-main\Real-time-Object-Detection-Flask-OpenCV-YoloV3-main\models\yolov3.cfg"
coco_path = "D:\BSCS 7\Machine Learning\Project\Real-time-Object-Detection-Flask-OpenCV-YoloV3-main\Real-time-Object-Detection-Flask-OpenCV-YoloV3-main\models\coco.names"
font = cv2.FONT_HERSHEY_PLAIN
distance_output = ''
classes = []  # loading all the object classes from coco.names to the classes variable
with open(coco_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
speak = Dispatch("SAPI.SpVoice").Speak
start = time.time()


# Load Yolo
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
cap = cv2.VideoCapture(0)




while True:
    ret, frame = cap.read()
    boxes = []
    class_ids = []
    confidences = []
    objectList= []
    type(frame)
    height, width, channels = frame.shape
    # detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, False)
    for b in blob:
        for n, img_blob in enumerate(b):
            # cv2.imshow(str(n),frame)
            n = 1
        
        
        
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Calculating confidence
    for out in outs:
         for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0:

                # detecting center of the rectangle
                cx = int(detection[0] * width)
                cy = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Formation of rectangle
                x = int(cx - w / 2)
                y = int(cy - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)
            if label in objectList:
                print("Already append")
            else:
                objectList.append(label)










# # image_processing_functions
# def process_frame(frame):
#     blob = cv2.dnn.blobFromImage(
#     frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

#     net.setInput(blob)
#     outs = net.forward(output_layers)
#     return outs

# def show_detected_object(frame, outs, width, height):
#     class_ids = []
#     confidences = []
#     boxes = []
#     objectList= []
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:
#                 # Object detected
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)

#                 # Rectangle coordinates
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)

#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

    # indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # for i in range(len(boxes)):
    #     if i in indexes:
    #         x, y, w, h = boxes[i]
    #         label = str(classes[class_ids[i]])
    #         color = colors[class_ids[i]]
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    #         cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)
    #         if label in objectList:
    #             print("Already append")
    #         else:
    #             objectList.append(label)
    # return objectList

# def FindObject(pipeline, objName, width = 640, height = 360):
#     frames = pipeline.wait_for_frames()
#     depth_frame = frames.get_depth_frame()
#     color_frame = frames.get_color_frame()
#     if not depth_frame or not color_frame:
#         return

#     distance = depth_frame.get_distance(int(width/2), int(height/2))
#     color_image = np.asanyarray(color_frame.get_data())

#     output = process_frame(color_image)
#     objectList= show_detected_object(color_image, output, width, height)
    

#     cv2.putText(color_image, "Dist: " + str(round(distance,2)),(30, 40), font, 3, (255, 0, 0), 3)

#     cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#     cv2.imshow('RealSense', color_image)
#     for i in range(len(objectList)):
#         # if (time.time() - start) > 30: 
#         #         speak("Cannot find {name}. Please re enter mode 2 again".format(name = objName))
#         #         return 'break'
#         if objectList[i] == objName:
#             if distance < 1:
#                 speak("Arrive")
#                 return 'break'
#             speak("{name} found {distance} meters in front of you. Move forward".format(name = objName,distance = round(distance,2)))
            
        


