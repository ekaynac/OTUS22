import torch
import threading

def run(model, im):
  results = model(im)
  results.save()

# Models
model0 = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=0)
model1 = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=1)

# Inference
threading.Thread(target=run, args=[model0, 'https://ultralytics.com/images/zidane.jpg'], daemon=True).start()
threading.Thread(target=run, args=[model1, 'https://ultralytics.com/images/bus.jpg'], daemon=True).start()