import torch
from PIL import Image
import os
import pathlib

pathlib.PosixPath = pathlib.WindowsPath

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best.pt')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)

def detect_text(image):
    # image ở đây là PIL Image object, không phải file
    if image.mode != 'RGB':
        image = image.convert('RGB')
    results = model(image)
    boxes = results.pandas().xyxy[0].to_dict(orient='records')
    return {"boxes": boxes, "status": "ok"}
