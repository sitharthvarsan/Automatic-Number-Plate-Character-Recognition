# 🚘 Automatic Number Plate Recognition (ANPR) System

**YOLOv8 + EasyOCR | Indian Number Plate Focus**

---

## 📌 Project Overview

This project implements a **real-time Automatic Number Plate Recognition (ANPR) system** designed specifically for **Indian vehicle number plates**.

The pipeline combines:

* **YOLOv8 (Nano)** for accurate and fast number plate detection
* **EasyOCR** for text recognition
* **Strict post-processing rules** based on Indian vehicle registration formats

The system supports:

* Image-based inference
* Webcam-based live capture
* Quantitative OCR evaluation against ground-truth data

---

## 🧠 High-Level Pipeline

```
Input Image / Webcam Frame
        ↓
YOLOv8 Number Plate Detection
        ↓
Plate Cropping
        ↓
Image Preprocessing (OCR-focused)
        ↓
EasyOCR Text Recognition
        ↓
Strict Indian Plate Post-Processing
        ↓
Final Plate Output / Evaluation Metrics
```

---

## 🧩 Pipeline Breakdown

### 1️⃣ Input Source

* Static car images
* Live webcam feed

Supported formats:

* `.jpg`, `.jpeg`, `.png`
* Webcam (OpenCV)

---

### 2️⃣ Number Plate Detection (YOLOv8)

* Model: **YOLOv8-nano**
* Task: Object detection (single class → `number_plate`)
* Trained on annotated Indian vehicle images

**Why YOLOv8-nano?**

* Lightweight
* Fast on CPU
* Sufficient accuracy for plate localization

**Output:**

* Bounding box coordinates of detected number plates

---

### 3️⃣ Plate Cropping

* Extracts the detected bounding box region from the image/frame
* Only the **plate region** is forwarded to OCR

This reduces noise and improves recognition accuracy.

---

### 4️⃣ Image Preprocessing for OCR

Applied only on the cropped plate region:

* Grayscale conversion
* Upscaling (2× using bicubic interpolation)
* Bilateral filtering (noise reduction while preserving edges)
* Histogram equalization (contrast enhancement)

**Purpose:**
Improve character clarity for OCR models.

---

### 5️⃣ Optical Character Recognition (EasyOCR)

* OCR Engine: **EasyOCR**
* Language: English (`en`)
* CPU-based inference

**Why EasyOCR?**

* Robust on distorted text
* Works well on number plates
* Easy integration

OCR returns:

* Detected text
* Confidence score

The highest-confidence result is selected.

---

### 6️⃣ Strict Indian Number Plate Post-Processing

This is the **core intelligence** of the system.

#### Enforced Format

```
AA00AA0000
││││││││││
│││││││└── Vehicle number (4 digits)
││││└──── Series letters (2 letters)
││└────── District code (2 digits)
└──────── State code (2 letters)
```

#### Key Features

* **State code validation** against all Indian states & UTs
* **Character-level OCR correction** (e.g., O↔0, I↔1, Z↔2)
* **State correction via candidate generation**, not hardcoding
* **Strict rejection** of invalid patterns

If the OCR result cannot be corrected into a valid Indian plate:
➡️ It is **rejected**, not force-fitted.

---

### 7️⃣ Final Output

* **Live webcam mode**

  * Press `C` to capture
  * OCR runs once
  * Final plate displayed on screen & terminal

* **Image mode**

  * Plate detection + recognition
  * Output image with bounding box & recognized plate

---

## 📊 OCR Evaluation Pipeline (`eval_ocr.py`)

Used for **quantitative evaluation** against ground-truth data.

### Evaluation Steps

1. Read validation images
2. Detect plate using YOLOv8
3. Run OCR + strict post-processing
4. Compare with ground truth CSV

### Metrics

* **Character Accuracy (%)**
* **Full Plate Accuracy (%)**

Rejected or undetected plates are **excluded** from unfair scoring.

---

## 📁 Project Structure

```
ANPR_YOLOv8/
│
├── data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   └── metadata/
│       └── Ground_Truth.csv
│
├── scripts/
│   ├── train_yolo.py
│   ├── infer_anpr.py
│   ├── eval_detection.py
│   └── eval_ocr.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Train YOLOv8

```bash
python scripts/train_yolo.py
```

### 3️⃣ Run Live ANPR (Webcam)

```bash
python scripts/infer_anpr.py
```

### 4️⃣ Evaluate OCR Accuracy

```bash
python scripts/eval_ocr.py
```

---

## ✅ Key Design Strengths

* ✔ Real-time capable (CPU)
* ✔ Strict domain-aware validation
* ✔ No hardcoded state assumptions
* ✔ Scalable to all Indian plates
* ✔ Production-style rejection logic
* ✔ Clean evaluation methodology

---

## 🚀 Future Improvements

* Multi-frame OCR voting (temporal smoothing)
* Night-time enhancement
* Motion blur handling
* GPU acceleration
* Deployment as REST API / Edge device

---

## 📌 Final Note

This project goes beyond basic ANPR demos by focusing on:

* **Domain correctness**
* **Robust post-processing**
* **Realistic evaluation**


