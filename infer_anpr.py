import cv2
from ultralytics import YOLO
import easyocr
import re
from itertools import product

# ===================== MODELS =====================

detector = YOLO(
    r"C:\Users\HP\OneDrive\Desktop\SITHARTH SNU\Project\ANPR_YOLOv8\runs\yolo_plate_ft\weights\best.pt"  # your YOLOv8 plate detector
)

ocr = easyocr.Reader(['en'], gpu=False)

# ===================== CONSTANTS =====================

VALID_STATE_CODES = {
    "AN","AP","AR","AS","BR","CG","CH","DD","DL","DN","GA",
    "GJ","HP","HR","JH","JK","KA","KL","LA","LD","MH","ML",
    "MN","MP","MZ","NL","OD","PB","PY","RJ","SK","TN","TR",
    "TS","UK","UP","WB"
}

def strict_indian_plate(raw_text):
    raw = ''.join(c for c in raw_text.upper() if c.isalnum())

    if len(raw) < 8:
        return ""

    raw = raw[:10].ljust(10, 'X')
    r = list(raw)

    # ---- STATE CODE (AA) ----
    for i in range(2):
        if not r[i].isalpha():
            r[i] = {
                '0': 'O', '1': 'I', '2': 'Z'
            }.get(r[i], 'M')

    state = ''.join(r[:2])
    if state not in VALID_STATE_CODES:
        return ""

    # ---- DISTRICT (00) ----
    for i in range(2, 4):
        if not r[i].isdigit():
            r[i] = {
                'O': '0', 'Q': '0',
                'I': '1', 'L': '1',
                'Z': '2'
            }.get(r[i], '0')

    # ---- SERIES (AA) ----
    for i in range(4, 6):
        if not r[i].isalpha():
            r[i] = {
                '0': 'O', '1': 'I', '2': 'Z',
                'Y': 'V'
            }.get(r[i], 'A')

    # ---- NUMBER (0000) ----
    for i in range(6, 10):
        if not r[i].isdigit():
            r[i] = {
                'O': '0', 'Q': '0',
                'I': '1', 'L': '1',
                'Z': '2', 'S': '5',
                'B': '8'
            }.get(r[i], '0')

    final = ''.join(r)

    if re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}', final):
        return final

    return ""

# ===================== IMAGE PREPROCESS =====================

def preprocess_plate_for_ocr(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.equalizeHist(gray)

    return gray

# ===================== WEBCAM CAPTURE =====================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("✅ Webcam ON")
print("👉 Press 'C' to CAPTURE plate")
print("👉 Press 'Q' or ESC to EXIT")

captured_frame = None
captured_box = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector(frame, conf=0.4)[0]

    if results.boxes is not None and len(results.boxes) > 0:
        box = results.boxes.xyxy[0]
        x1, y1, x2, y2 = map(int, box.tolist())

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Press 'C' to capture",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("ANPR - Live Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and results.boxes is not None and len(results.boxes) > 0:
        print("\n📸 Frame captured. Closing webcam...")
        captured_frame = frame.copy()
        captured_box = results.boxes.xyxy[0]
        break

    if key == 27 or key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# ===================== OCR + POSTPROCESS =====================

x1, y1, x2, y2 = map(int, captured_box.tolist())
plate_crop = captured_frame[y1:y2, x1:x2]

processed_plate = preprocess_plate_for_ocr(plate_crop)

print("🔍 Running OCR...")

ocr_results = ocr.readtext(processed_plate, detail=1)
raw_text = ""

if ocr_results:
    ocr_results.sort(key=lambda x: x[2], reverse=True)
    raw_text = ocr_results[0][1]

final_text = strict_indian_plate(raw_text)

# ===================== TERMINAL OUTPUT =====================

print("===================================")
print("✅ FINAL RECOGNIZED NUMBER PLATE")
print("➡️ RAW OCR   :", raw_text)
print("➡️ FINAL     :", final_text if final_text else "[REJECTED]")
print("===================================\n")

# ===================== DISPLAY RESULT =====================

cv2.rectangle(captured_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
cv2.putText(
    captured_frame,
    f"PLATE: {final_text}",
    (x1, y1 - 15),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (0, 0, 255),
    3
)

cv2.imshow("ANPR - Final Result", captured_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
