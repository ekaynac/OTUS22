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
model.conf = 0.5 # NMS confidence threshold
#model.iou = 0.45  # NMS IoU threshold
#model.agnostic = False  # NMS class-agnostic
#model.multi_label = False  # NMS multiple labels per box
#model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
model.max_det = 100 # maximum number of detections per image
#model.amp = False  # Automatic Mixed Precision (AMP) inference

# Standarts for detection 
XMIN = int(XSIZE/4)
YMIN = int(YSIZE/10)
XMAX = int(3*XSIZE/4)
YMAX = int(9*YSIZE/10)
margin = 100 # %marginTOexceedDetectionWindow

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
cap = cv2.VideoCapture(r"D:\Github\Distance-Measurement\deltaw.mp4")

# Define the codec and create VideoWriter object
#fourcc = cv2.cv.CV_FOURCC(*'DIVX')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
out = cv2.VideoWriter('outputShahed.mp4', -1, 60.0, (XSIZE,YSIZE))

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (XSIZE,YSIZE), interpolation = cv2.INTER_AREA)
    
    results = model(frame)
    frame = cv2.rectangle(frame, (XMIN, YMIN), ( XMAX, YMAX), color, 2)
    # Results
    df = results.pandas().xyxy[0]

    if(len(df["xmin"])>0):
        for i in range(len(df["xmin"])):
                if(((df["xmax"][i] - df["xmin"][i])/XSIZE > 0.00 or (df["ymax"][i] - df["ymin"][i])/YMAX > 0.00)):
                    if (df["xmax"][i]<=XMAX*(100+margin)/100 and df["ymax"][i]<=YMAX*(100+margin)/100 and df["xmin"][i]>=XMIN*(100+margin)/100 and df["ymin"][i]>=YMIN*(100+margin)/100):
                        frame = cv2.rectangle(frame, (int(df["xmin"][i]),int(df["ymin"][i])), (int(df["xmax"][i]),int(df["ymax"][i])), color, thickness)
                        frame = cv2.putText(frame, 'Locked:{:.2f}'.format(df["confidence"][i]), (int(df["xmin"][i]),int(df["ymin"][i])), font, 
                        fontScale, color, thickness, cv2.LINE_AA)
                        w= int(df["xmin"][i] + int((df["xmax"][i] - df["xmin"][i])/2))
                        h= int(df["ymin"][i] + int((df["ymax"][i] - df["ymin"][i])/2))
                        frame = cv2.line(frame, (int(XSIZE/2),int(YSIZE/2)) ,(w,h), (0,0,255), 1)
                        print("Kilitlendi")
                    else:
                        print("Tespit var ama belirlenen alanda değil")
                        frame = cv2.rectangle(frame, (int(df["xmin"][i]),int(df["ymin"][i])), (int(df["xmax"][i]),int(df["ymax"][i])), (255,0,0), thickness)
                else:
                    frame = cv2.rectangle(frame, (int(df["xmin"][i]),int(df["ymin"][i])), (int(df["xmax"][i]),int(df["ymax"][i])), (0,255,0), thickness)
                    print("Standart dışı tespit var")
    else:
        frame = cv2.putText(frame, 'No detection', org, font, 
            fontScale, color, thickness, cv2.LINE_AA)

    out.write(frame)
    
    cv2.imshow("Inference", frame)

    key = cv2.waitKey(1)
    if key==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

