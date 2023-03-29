import cv2
import os
import torch

def xyxyTOyolo(xmin, ymin, xmax, ymax, shapex,shapey):
    w=  (xmax-xmin)/shapex
    h=  (ymax-ymin)/shapey
    xc= xmin/shapex + w/2
    yc= ymin/shapey + h/2
    return xc, yc, w, h

# Model weight path
MODELPATH = r'D:\Github\OTUS22\Weights\FinalUAV\best.pt'

# Model
model = torch.hub.load(r'D:\Github\OTUS22\yolov5', 'custom',path=MODELPATH, source='local', force_reload=True)

# Inference
XSIZE=640
YSIZE=480
model.conf = 0.2 # NMS confidence threshold
#model.iou = 0.45  # NMS IoU threshold
#model.agnostic = False  # NMS class-agnostic
#model.multi_label = False  # NMS multiple labels per box
#model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
model.max_det = 100 # maximum number of detections per image
#model.amp = False  # Automatic Mixed Precision (AMP) inference
loosyflag=1

# Standarts for detection 
XMIN = int(XSIZE/10)
YMIN = int(YSIZE/4)
XMAX = int(9*XSIZE/10)
YMAX = int(3*YSIZE/4)
margin = 5 # %marginTOexceedDetectionWindow

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
cap = cv2.VideoCapture(r"D:\Github\OTUS22\başarılı.mp4")

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (XSIZE,YSIZE), interpolation = cv2.INTER_AREA)
    frame = cv2.rectangle(frame, (XMIN, YMIN), ( XMAX, YMAX), color, 2)
    results = model(frame)
    # Results
    df = results.pandas().xyxy[0]

    if(len(df["xmin"])>0):
        for i in range(len(df["xmin"])):
            if (df["xmax"][i]<=XMAX*(100+margin)/100 and df["ymax"][i]<=YMAX*(100+margin)/100 and df["xmin"][i]>=XMIN*(100+margin)/100 and df["ymin"][i]>=YMIN*(100+margin)/100):
                if(((df["xmax"][i] - df["xmin"][i])/XSIZE > 0.06 or (df["ymax"][i] - df["ymin"][i])/YMAX > 0.06)):
                    frame = cv2.rectangle(frame, (int(df["xmin"][i]),int(df["ymin"][i])), (int(df["xmax"][i]),int(df["ymax"][i])), color, thickness)
                    frame = cv2.putText(frame, '{}:{:.2f}'.format(df["class"][i],df["confidence"][i]), (int(df["xmin"][i]),int(df["ymin"][i])), font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    else:
        frame = cv2.putText(frame, 'No detection', org, font, 
            fontScale, color, thickness, cv2.LINE_AA)

        
    cv2.imshow("sex", frame)

    key = cv2.waitKey(1)
    if key==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

