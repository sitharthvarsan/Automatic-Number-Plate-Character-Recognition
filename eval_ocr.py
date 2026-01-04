import cv2
import pandas as pd
import re
from ultralytics import YOLO
import easyocr
from itertools import product

# ================= CONFIG =================

IMG_DIR = r"C:\Users\HP\OneDrive\Desktop\SITHARTH SNU\Project\ANPR_YOLOv8\data\images\val"
CSV_PATH = r"C:\Users\HP\OneDrive\Desktop\SITHARTH SNU\Project\ANPR_YOLOv8\data\metadata\Ground_Truth.csv"
MODEL_PATH = r"C:\Users\HP\runs\detect\train5\weights\best.pt"

# ================= MODELS =================

detector = YOLO(MODEL_PATH)
ocr = easyocr.Reader(['en'], gpu=False)

# ================= CONSTANTS =================

VALID_STATE_CODES = {
    "AN","AP","AR","AS","BR","CG","CH","DD","DL","DN","GA",
    "GJ","HP","HR","JH","JK","KA","KL","LA","LD","MH","ML",
    "MN","MP","MZ","NL","OD","PB","PY","RJ","SK","TN","TR",
    "TS","UK","UP","WB"
}

CHAR_CONFUSIONS = {
    'H': ['H', 'M', 'N'],
    'M': ['M', 'H'],
    'N': ['N', 'M'],
    'O': ['O', '0'],
    'Q': ['0'],
    'I': ['I', '1'],
    'L': ['L', '1'],
    'Z': ['Z', '2'],
    'S': ['S', '5'],
    'B': ['B', '8'],
}

# ================= UTILS =================

def clean_text(text):
    return ''.join(c for c in text.upper() if c.isalnum())


def correct_state_code(state):
    state = state.upper()
    candidates = []

    for ch in state:
        candidates.append(CHAR_CONFUSIONS.get(ch, [ch]))

    for comb in product(*candidates):
        candidate = ''.join(comb)
        if candidate in VALID_STATE_CODES:
            return candidate

    return ""


def strict_indian_plate(raw_text):
    raw = clean_text(raw_text)

    if len(raw) < 8:
        return ""

    raw = raw[:10].ljust(10, 'X')
    result = list(raw)

    # STATE
    state = correct_state_code(''.join(result[:2]))
    if not state:
        return ""
    result[0], result[1] = state

    # DISTRICT
    for i in range(2, 4):
        if not result[i].isdigit():
            result[i] = {'O':'0','Q':'0','I':'1','L':'1','Z':'2'}.get(result[i], '0')

    # SERIES
    for i in range(4, 6):
        if not result[i].isalpha():
            result[i] = {'0':'O','1':'I','2':'Z'}.get(result[i], 'A')

    # NUMBER
    for i in range(6, 10):
        if not result[i].isdigit():
            result[i] = {
                'O':'0','Q':'0','I':'1','L':'1',
                'Z':'2','S':'5','B':'8'
            }.get(result[i], '0')

    final_plate = ''.join(result)

    if re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}', final_plate):
        return final_plate

    return ""


def preprocess_plate_for_ocr(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    gray = cv2.resize(gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.equalizeHist(gray)
    return gray


def char_accuracy(pred, gt):
    matches = sum(p == g for p, g in zip(pred, gt))
    return matches / max(len(gt), 1)

# ================= EVALUATION =================

df = pd.read_csv(CSV_PATH)

total_chars = 0
correct_chars = 0
full_match = 0
evaluated = 0

for _, row in df.iterrows():
    img_path = f"{IMG_DIR}/{row['Image']}"
    gt_text = clean_text(row['License_Plate'])

    image = cv2.imread(img_path)
    if image is None:
        continue

    results = detector(image, conf=0.4)[0]
    if results.boxes is None or len(results.boxes) == 0:
        continue

    box = results.boxes.xyxy[0]
    x1, y1, x2, y2 = map(int, box.tolist())
    plate = image[y1:y2, x1:x2]

    processed = preprocess_plate_for_ocr(plate)
    ocr_res = ocr.readtext(processed, detail=1)

    raw_text = ""
    if ocr_res:
        ocr_res.sort(key=lambda x: x[2], reverse=True)
        raw_text = ocr_res[0][1]

    pred_text = strict_indian_plate(raw_text)

    if not pred_text:
        continue  # rejected plates not counted

    evaluated += 1

    if pred_text == gt_text:
        full_match += 1

    correct_chars += char_accuracy(pred_text, gt_text) * len(gt_text)
    total_chars += len(gt_text)

# ================= RESULTS =================

char_acc = (correct_chars / total_chars) * 100 if total_chars else 0
plate_acc = (full_match / evaluated) * 100 if evaluated else 0

print("\n===== OCR EVALUATION RESULTS =====")
print("Evaluated Images      :", evaluated)
print(f"Character Accuracy    : {char_acc:.2f}%")
print(f"Full Plate Accuracy   : {plate_acc:.2f}%")
