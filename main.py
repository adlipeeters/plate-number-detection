# This is a sample Python script.
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('ultralytics/runs/detect/train_model/weights/best.pt')

# Run inference on 'bus.jpg' with arguments
model.predict('assets/videos/1.mp4', save=True, imgsz=320, conf=0.2)

