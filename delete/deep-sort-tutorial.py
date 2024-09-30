#Necessary imports
import cv2
import numpy as np
import sys
import glob
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import torch


class YoloDetector():

    def __init__(self):
        #Using yolov5s for our purposes of object detection, you may use a larger model
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained = True)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5s.pt', force_reload=False)

        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using Device: ', self.device)
    
    def score_frame(self, frame):
        self.model.to(self.device)
        downscale_factor = 2
        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame = cv2.resize(frame, (width, height))

        results = self.model(frame)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        return labels, cord
    
    def class_to_label(self, x):
        return self.classes[int(x)]
    
    def plot_boxes(self, results, frame, height, width, confidence=0.3):

        labels, cord = results
        detections = []

        n = len(labels)
        x_shape, y_shape = width, height

        for i in range(n):
            row = cord[i]

            if row[4]>=confidence:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                
                #In this demonstration, we will only be detecting persons. You can add classes of your choice
                if self.class_to_label(labels[i]) == 'car':

                    x_center = x1 + (x2-x1)
                    y_center = y1 + ((y2-y1) / 2)

                    tlwh = np.asarray([x1, y1, int(x2-x1), int(y2-y1)], dtype = np.float32)
                    confidence = float(row[4].item())
                    feature = 'car'

                    detections.append(([x1, y1, int(x2-x1), int(y2-y1)], row[4].item(), 'car'))
        
        return frame, detections
    
# Setting input video to webcam
# To use your own pre-downloaded videos, write the file path instead of 0
# cap = cv2.VideoCapture(0)
source = './videos/park-1.mp4'
cap = cv2.VideoCapture(source)

# Setting resolution for webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#Initializing the detection class
detector = YoloDetector()

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#Initialise the object tracker class
object_tracker = DeepSort()

while cap.isOpened():
    success, img = cap.read()

    start = time.perf_counter()

    results = detector.score_frame(img)
    img,detections = detector.plot_boxes(results, img, height=img.shape[0], width=img.shape[1], confidence=0.5)

    tracks = object_tracker.update_tracks(detections, frame=img) 
    # NOTE: Bounding box expects to be a list of detections, each in tuples of ([left, top, w, h], confidence, detection class)
    
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()

        bbox = ltrb

        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255),2)
        cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]),int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

    end = time.perf_counter()
    totalTime = end-start
    fps = 1/totalTime

    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()

cv2.destroyAllWindows()