import cv2
import numpy as np
from gtts import gTTS
language = 'en'

# loading yolo algorithm
net = cv2.dnn.readNet("models/yolov3.weights", "models/yolov3.cfg")
print(net)
classes = []
with open("models/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
# Reading  layers`
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
cap = cv2.VideoCapture(0)
# loading the image
while True:
    ret, frame = cap.read()
    boxes = []
    class_ids = []
    confidences = []
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
                
                
            
            
            
            
       
            # cv2.rectangle(img,(x,y),((x+w),(y+h)),(0,255,0),2)
    font = cv2.FONT_HERSHEY_PLAIN
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y + 30), font, 2, (255, 0, 255), 2)
            myobj = gTTS(text=label, lang=language, slow=False)
            myobj.save("welcome.mp3")

    cv2.imshow("Image", frame)
    if cv2.waitKey(20) & 0xFF == ord(" "):
        exit(0)