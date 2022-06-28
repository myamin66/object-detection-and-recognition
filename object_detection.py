import time
import cv2
import numpy as np
import cv2
import torch


import pyttsx3
import math
from cvzone import HandTrackingModule
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
import gtts
from playsound import playsound


class ObjectDetection:
    def __init__(self):
        self.MODEL = cv2.dnn.readNet(
            'models/yolov3.weights',
            'models/yolov3.cfg'
        )

        self.CLASSES = []
        with open("models/coco.names", "r") as f:
            self.CLASSES = [line.strip() for line in f.readlines()]

        self.OUTPUT_LAYERS = [self.MODEL.getLayerNames()[i - 1]
                              for i in self.MODEL.getUnconnectedOutLayers()]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        self.COLORS /= (np.sum(self.COLORS**2, axis=1)**0.5/255)[np.newaxis].T

    def detectHand(self,snap):
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        mpDraw = mp.solutions.drawing_utils


        # Load the gesture recognizer model
        hand_model = load_model('mp_hand_gesture')
        detector = HandTrackingModule.HandDetector()

        # Load class names for hand gestures
        f = open('gesture.names', 'r')
        classNames1 = f.read().split('\n')
        f.close()
        frame = snap

        x, y, c = frame.shape
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        Detected_hand_class = ''

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                prediction = hand_model.predict([landmarks])
                classID = np.argmax(prediction)
                Detected_hand_class = classNames1[classID]
        if(Detected_hand_class == 'Empty' or Detected_hand_class == 'Empty hand'):
            cv2.putText(frame, Detected_hand_class, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)
            
            engine = pyttsx3.init()
              
            engine.setProperty('rate', 125)
            engine.setProperty('volume',1.0) 
            voices = engine.getProperty('voices')
            engine.setProperty('voice', voices[1].id)
            engine.say("the hand is Empty")
            engine.runAndWait()





        return frame



    def detectObj(self, snap):
        height, width, channels = snap.shape
        blob = cv2.dnn.blobFromImage(
            snap, 1/255, (416, 416), swapRB=True, crop=False)

        self.MODEL.setInput(blob)
        outs = self.MODEL.forward(self.OUTPUT_LAYERS)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    # Rectangle coordinates
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                engine = pyttsx3.init()              
                engine.setProperty('rate', 125)
                engine.setProperty('volume',1.0) 
                voices = engine.getProperty('voices')
                engine.setProperty('voice', voices[1].id)
                label = str(self.CLASSES[class_ids[i]])
                engine.say("There is " + label + " in the video")
                engine.runAndWait()

                color = self.COLORS[i]
                cv2.rectangle(snap, (x, y), (x + w, y + h), color, 2)
                cv2.putText(snap, label, (x, y - 5), font, 2, color, 2)
        return snap
        
class VideoStreaming(object):
    def __init__(self):
        super(VideoStreaming, self).__init__()
        self.VIDEO = cv2.VideoCapture(0)

        self.MODEL = ObjectDetection()
        
        self._preview = True
        self._hand = False
        self._flipH = False
        self._detect = False
        self._exposure = self.VIDEO.get(cv2.CAP_PROP_EXPOSURE)
        self._contrast = self.VIDEO.get(cv2.CAP_PROP_CONTRAST)
    @property
    def preview(self):
        return self._preview
    @preview.setter
    def preview(self, value):
        self._preview = bool(value)

        
    

    
    

    @property
    def flipH(self):
        return self._flipH

    @flipH.setter
    def flipH(self, value):
        self._flipH = bool(value)

    @property
    def detect(self):
        return self._detect

    @detect.setter
    def detect(self, value):
        self._detect = bool(value)


    @property
    def hand(self):
        return self._hand

    @hand.setter
    def hand(self, value):
        self._hand = bool(value)

    @property
    def exposure(self):
        return self._exposure

    @exposure.setter
    def exposure(self, value):
        self._exposure = value
        self.VIDEO.set(cv2.CAP_PROP_EXPOSURE, self._exposure)

    @property
    def contrast(self):
        return self._contrast

    @contrast.setter
    def contrast(self, value):
        self._contrast = value
        self.VIDEO.set(cv2.CAP_PROP_CONTRAST, self._contrast)

    def show(self):
        while(self.VIDEO.isOpened()):
            ret, snap = self.VIDEO.read()
            if self.flipH:
                snap = cv2.flip(snap, 1)

            if ret == True:
                if self._preview:
                    if self._hand:
                        snap = self.MODEL.detectHand(snap)
                    elif self.detect:
                        snap = self.MODEL.detectObj(snap)

                else:
                    snap = np.zeros((
                        int(self.VIDEO.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        int(self.VIDEO.get(cv2.CAP_PROP_FRAME_WIDTH))
                    ), np.uint8)
                    label = 'camera disabled'
                    H, W = snap.shape
                    font = cv2.FONT_HERSHEY_COMPLEX
                    color = (255, 255, 255)
                    cv2.putText(snap, label, (W//2, H//2), font, 2, color, 2)

                frame = cv2.imencode('.jpg', snap)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.01)

            else:
                break
        print('off')
