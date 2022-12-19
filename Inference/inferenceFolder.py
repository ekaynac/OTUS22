import torch
import cv2
import os

PATH = "F:/SabitKanatVideolar/Labellanacakvideolar/GH010012_Trim/"
MODELPATH = "D:/Github/OTUS22/Weights/UAVweights300/best.onnx"

def drawBox(img,bbox):
    x,y,w,h = tuple(map(int,bbox))
    cv2.rectangle(img,(x,y),((x+w),(y+h)),(255,0,255),3,1)
    cv2.putText(img, "Tracking", (75, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODELPATH)  # or yolov5n - yolov5x6, custom

# Inference
model.conf = 0.6 # NMS confidence threshold
#model.iou = 0.45  # NMS IoU threshold
#model.agnostic = False  # NMS class-agnostic
#model.multi_label = False  # NMS multiple labels per box
#model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
#model.max_det = 1000  # maximum number of detections per image
#model.amp = False  # Automatic Mixed Precision (AMP) inference

# Options for cv2.rectangle
thickness=1
color=(0,0,255)

# Options for cv2.putText
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 0.5

for i in os.listdir(PATH):
    try:
        frame = cv2.imread(PATH + i)
        frame = cv2.resize(frame, (640,640), interpolation = cv2.INTER_AREA)
        
        results = model(frame)
        # Results
        df = results.pandas().xyxy[0]
        try:
            frame = cv2.rectangle(frame, (int(df["xmin"]),int(df["ymin"])), (int(df["xmax"]),int(df["ymax"])), color, thickness)
            frame = cv2.putText(frame, 'UAV: {}'.format(df["confidence"]), (int(df["xmin"]),int(df["ymin"])), font, 
                    fontScale, color, thickness, cv2.LINE_AA)
        except:
            frame = cv2.putText(frame, 'No detection', org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow("sex", frame)

        key = cv2.waitKey(1)
        if key==ord('q'):
            break
    except:
        pass
cv2.destroyAllWindows()