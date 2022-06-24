import pickle
import json
import sys
import cv2
import pyrealsense2 as rs
import cv2
from pathlib import Path
import numpy as np
from win32com.client import Dispatch
import time

import numpy as np

from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import img_to_array

from PIL import Image
import requests
from io import BytesIO

import traceback
from win32com.client import Dispatch

#constant
speak = Dispatch("SAPI.SpVoice").Speak

def init():
    global caption_model
    global tokenizer
    global encode_model
    global model_path

    global MAX_LEN
    global OUTPUT_DIM
    global WIDTH
    global HEIGHT

    MAX_LEN = 51
    OUTPUT_DIM = 2048
    WIDTH = 299
    HEIGHT = 299

    caption_model = load_model("../model/caption_model.h5")
    encode_model = load_model("../model/encode_model.h5")

    with open('../model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)


def describeImage(img_url):
    try:
        caption = "startseq"

        img = Image.open(img_url)
        img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
        img = img_to_array(img)
        # img = img_url
        # preprocess the image

        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        x1 = encode_model.predict(img)
        x1 = x1.reshape((1, OUTPUT_DIM))

        # generate the caption

        for i in range(MAX_LEN):
            seq = tokenizer.texts_to_sequences([caption])
            x2 = pad_sequences(seq, maxlen=MAX_LEN)

            y = caption_model.predict([x1, x2], verbose=0)
            word = tokenizer.index_word[np.argmax(y)]

            if word == "endseq":
                break

            caption += " " + word

        caption = caption.replace("startseq", "").strip()
        speak(caption)
        return {"caption": caption}

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}

# cap = cv2.VideoCapture(0)
def describe_video_stream(pipeline):
    init()
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return

    # img = Image.open(color_frame.get_data())
    # img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
    # img = img_to_array(img)

    color_image = np.asarray(color_frame.get_data())
    im = Image.fromarray(color_image)
    im.save("mode3.jpg")
    describeImage("mode3.jpg")




if __name__ == '__main__':
    init()
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
        
        # depth_frame = frame.get_depth_frame()
        # color_frame = frame.get_color_frame()
        # if not depth_frame or not color_frame:
        #     break
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

                    # detecting center of the rectangle
                    
        
        font = cv2.FONT_HERSHEY_PLAIN
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
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

        cv2.imshow("Image" , frame)
        if cv2.waitKey(20) & 0xFF == ord(" "):
            exit(0)
        # color_image = np.asarray(frame.get_data())
        im = Image.fromarray(frame)
        im.save("mode3.jpg")
        describeImage("mode3.jpg")






