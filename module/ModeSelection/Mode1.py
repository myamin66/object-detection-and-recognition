import pyrealsense2 as rs
import cv2
import numpy as np
import os
from win32com.client import Dispatch
import sys

# constant
font = cv2.FONT_HERSHEY_PLAIN
distance_output = ''
speak = Dispatch("SAPI.SpVoice").Speak


def Beep(direction):
    if direction == 'front':
        speak("front")
    if direction == 'right':
        speak("right")
    if direction == 'left':
        speak("left")


def ObjectAvoidance(pipeline, width=640, height=360):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return

    # Left search
    minDepthLeft = width
    minXLeft = 0
    step = 5

    leftThreshold = int(width / 6)

    for x in range(leftThreshold, int(width / 2), step):
        for y in range(0, int(height), step):
            dist = depth_frame.get_distance(x, y)
            if (dist != 0 and dist < minDepthLeft):
                minDepthLeft = dist
                minXLeft = x

    # Right search
    minDepthRight = width
    minXRight = 0

    rightThreshold = int(width * 5 / 6)
    for x in range(int(width / 2), rightThreshold, step):
        for y in range(0, int(height), step):
            dist = depth_frame.get_distance(x, y)
            if (dist != 0 and dist < minDepthRight):
                minDepthRight = dist
                minXRight = x

    depthThreshold = 0.6  # detec objects within 0.6 meters

    if minDepthRight < depthThreshold and minDepthRight != 0.0:
        if minXRight < (width * 3 / 4):
            distance_output = "Front: " + str(round(minDepthRight, 2))
            Beep('front')
        else:
            distance_output = "Right: " + str(round(minDepthRight, 2))
            Beep('right')

    elif minDepthLeft < depthThreshold and minDepthLeft != 0.0:
        if minXLeft > (width / 4):
            distance_output = "Front: " + str(round(minDepthLeft, 2))
            Beep('front')
        else:
            distance_output = "Left: " + str(round(minDepthLeft, 2))
            Beep('left')
    else:
        distance_output = "Front: " + str(round(minDepthLeft, 2))

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Writing out distance info on screen
    if distance_output != '':
        cv2.putText(color_image, distance_output, (30, 40), font, 3, (255, 0, 0), 3)

    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', color_image)
