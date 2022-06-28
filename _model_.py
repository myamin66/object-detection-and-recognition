# TechVidvan hand Gesture Recognizer

# import necessary packages
import cv2
import torch


import pyttsx3
import math
from cvzone import HandTrackingModule
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import gtts
from playsound import playsound

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


# Load the gesture recognizer model
hand_model = load_model('mp_hand_gesture')
detector = HandTrackingModule.HandDetector()


#loding yolo model
object_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


# Load class names for hand gestures
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()



# Initialize the webcam
cap = cv2.VideoCapture(0)





while True:
    # Read each frame from the webcam
    _, frame = cap.read()
    dummy = frame
    results = object_model(frame)
    #,_=  detector.findHands(dummy) #for bounding box of hand
    ds = results.pandas().xyxy[0]
   
    x, y, c = frame.shape
    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)
    Detected_hand_class = ''
   

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = hand_model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            Detected_hand_class = classNames[classID]

    # show the prediction on the frame
    if(Detected_hand_class == 'Empty' or Detected_hand_class == 'Empty hand'):
        cv2.putText(frame, Detected_hand_class, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    
   
    cv2.imshow("Output", frame) 
    #if (hand):
    if(Detected_hand_class == 'Empty' or Detected_hand_class == 'Empty hand'): 
        
      engine = pyttsx3.init()
      engine.say("Hand is empty")
      engine.setProperty('rate', 125)
      engine.setProperty('volume',1.0) 
      voices = engine.getProperty('voices')
      engine.setProperty('voice', voices[1].id)
      engine.runAndWait()
   
    else:
        
      ##If hand is not empty
    
      for index in ds.index:
          if(ds.loc[index,'name'] != 'person' and ds.loc[index,'name'] != 'clock' and ds.loc[index,'name'] != 'chair' ):
              #print("Detected hand", className, "boxValue",boxValue)
              xmin,xmax,ymin,ymax =int(ds.iloc[index].xmin), int(ds.iloc[index].ymin), int(ds.iloc[index].xmax), int(ds.iloc[index].ymax)
              print("Detected object boundings", xmin," ",ymin," ",xmax," ",ymax)
              engine = pyttsx3.init()
              
              engine.setProperty('rate', 125)
              engine.setProperty('volume',1.0) 
              voices = engine.getProperty('voices')
              engine.setProperty('voice', voices[1].id)
              engine.say("There is " + ds.loc[index,'name'] + "in the hand")
              engine.runAndWait()
              
       
     
       
    

    if cv2.waitKey(1) == ord('q'):
        break
 

  



# release the webcam and destroy all active windows





cap.release()

cv2.destroyAllWindows()