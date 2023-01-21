import cv2
import os
import torch

# Model weight path
MODELPATH = '/home/otus/Desktop/enesbozuyor/OTUS22-master/Weights/FinalUAV/best16.engine'

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom',path=MODELPATH)  # or yolov5n - yolov5x6, custom

# Inference
model.conf = 0.2 # NMS confidence threshold
#model.iou = 0.45  # NMS IoU threshold
#model.agnostic = False  # NMS class-agnostic
#model.multi_label = False  # NMS multiple labels per box
#model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
model.max_det = 2  # maximum number of detections per image
#model.amp = False  # Automatic Mixed Precision (AMP) inference

def drawBox(img,bbox):
    x,y,w,h = tuple(map(int,bbox))
    cv2.rectangle(img,(x,y),((x+w),(y+h)),(255,0,255),3,1)
    cv2.putText(img, "Tracking", (75, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def createTracker(num):
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[num]

    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create() 
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create() 
    if tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create() 
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create() 
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    return tracker

# Options for cv2.rectangle
thickness=1
color=(0,0,255)

# Options for cv2.putText
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 0.5

# Video address
VIDPATH = 0

# Video Capturer
cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink")

cnt = 0
init = 1
TRACKERTYPE = 2
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640,640), interpolation = cv2.INTER_AREA)
    if ret:
        cnt+=1
        if cnt == 1:
            cnt = 0
            results = model(frame)
            df = results.pandas().xyxy[0]
            df["xmin"] = df["xmin"].astype(int)
            df["ymin"] = df["ymin"].astype(int)
            df["xmax"] = df["xmax"].astype(int)
            df["ymax"] = df["ymax"].astype(int)
            if df.size != 0:
                for i in range(df.shape[0]):
                    frame = cv2.rectangle(frame, (df["xmin"][i],df["ymin"][i]), (df["xmax"][i],df["ymax"][i]), color, thickness)
                    frame = cv2.putText(frame, 'UAV: {}'.format(df["confidence"][i]), (df["xmin"][i],df["ymin"][i]), font, 
                    fontScale, color, thickness, cv2.LINE_AA)
                    bbox= (df["xmin"][0], df["ymin"][0], df["xmax"][0]- df["xmin"][0], df["ymax"][0]- df["ymin"][0])
                    tracker = createTracker(TRACKERTYPE)
                    _ = tracker.init(frame, bbox)
        
        try:
            _, bbox = tracker.update(frame)
            drawBox(frame,bbox)
        except:
            frame = cv2.putText(frame, 'No detection', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow("sex", frame)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    key = cv2.waitKey(1)
    if key==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()
