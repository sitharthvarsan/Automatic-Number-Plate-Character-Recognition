from ultralytics import YOLO
import os

# Absolute paths (NO ambiguity)
PROJECT_ROOT = r"C:\Users\HP\OneDrive\Desktop\SITHARTH SNU\Project\ANPR_YOLOv8"
DATA_YAML = os.path.join(PROJECT_ROOT, "anpr.yaml")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

model = YOLO("yolov8n.pt")

model.train(
    data=DATA_YAML,
    epochs=50,               # ✅ GOOD for CPU
    batch=8,                 # ✅ Safe for i5 CPU
    imgsz=640,
    patience=15,
    device="cpu",
    workers=4,
    optimizer="AdamW",
    lr0=0.003,
    pretrained=True,
    project=RUNS_DIR,
    name="yolo_plate_ft",
    exist_ok=True
)
