import os
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- IOU ----------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\Users\HP\OneDrive\Desktop\SITHARTH SNU\Project\ANPR_YOLOv8\runs\yolo_plate_ft\weights\best.pt"
IMG_DIR = r"C:\Users\HP\OneDrive\Desktop\SITHARTH SNU\Project\ANPR_YOLOv8\data\Images\Val"
LBL_DIR = r"C:\Users\HP\OneDrive\Desktop\SITHARTH SNU\Project\ANPR_YOLOv8\data\Labels\Val"

model = YOLO(MODEL_PATH)

correct = 0
total = 0

for img_name in os.listdir(IMG_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMG_DIR, img_name)
    lbl_path = os.path.join(LBL_DIR, os.path.splitext(img_name)[0] + ".txt")

    if not os.path.exists(lbl_path):
        continue

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Ground truth
    with open(lbl_path) as f:
        cls, xc, yc, bw, bh = map(float, f.readline().split())

    gt_box = [
        int((xc - bw/2) * w),
        int((yc - bh/2) * h),
        int((xc + bw/2) * w),
        int((yc + bh/2) * h),
    ]

    results = model(img)[0]
    if results.boxes is None:
        total += 1
        continue

    preds = results.boxes.xyxy.cpu().numpy()

    total += 1
    for p in preds:
        if iou(gt_box, p) >= 0.5:
            correct += 1
            break

print("\n===== DETECTION EVALUATION =====")
print(f"Total Images     : {total}")
print(f"Correct Detections (IoU≥0.5): {correct}")
print(f"Detection Accuracy: {(correct/total)*100:.2f}%")
