# 🇮🇳 Indian License Plate Recognition (YOLOv8 + Fine-Tuned PaddleOCR)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8n-green)
![PaddleOCR](https://img.shields.io/badge/Recognition-PaddleOCR%20(Fine--Tuned)-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

An end-to-end **Automatic Number Plate Recognition (ANPR)** system optimized for Indian vehicles. This project integrates **YOLOv8 Nano** for real-time plate detection and a **Fine-Tuned PaddleOCR** model (SVTR_LCNet) for high-accuracy text recognition.

It features a unique **"Burst Mode"** with temporal voting to eliminate OCR flickering and applies strict **Indian Syntax Constraints** to correct common OCR errors (e.g., misreading 'O' as 'D' or '8' as 'B').

---

## 🚀 Key Features

* **🏎️ Real-Time Detection:** Uses `YOLOv8-Nano`, the lightest and fastest model, achieving real-time performance on CPU.
* **🧠 Fine-Tuned OCR:** Custom-trained **PaddleOCR (SVTR_LCNet)** model on Indian license plate datasets, significantly outperforming generic OCR models on Indian fonts.
* **📸 Burst Mode (Temporal Voting):** Captures **15 consecutive frames** and performs statistical voting to stabilize character predictions.
* **🇮🇳 Indian Format Logic:** Auto-corrects characters based on the standard format `AA 00 AA 0000`:
    * *State Code (First 2 chars)* → Forced to be Letters (e.g., `8G` → `MH`).
    * *District Code (Next 2 chars)* → Forced to be Digits (e.g., `O1` → `01`).

---

## 🛠️ Tech Stack

* **Detection:** [Ultralytics YOLOv8 (Nano)](https://github.com/ultralytics/ultralytics)
* **Recognition:** [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Fine-Tuned on Colab)
* **Processing:** OpenCV, NumPy
* **Language:** Python 3.10+

---

## 📂 Project Structure

```text
.
├── inference/                  # Fine-tuned PaddleOCR weights
│   ├── inference.pdmodel       # Model architecture
│   └── inference.pdiparams     # Learned weights
├── runs/detect/train/weights/  # Trained YOLOv8 model (best.pt)
├── scripts/
│   └── infer_anpr.py           # Main inference script
├── en_dict.txt                 # Character dictionary for PaddleOCR
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```
⚙️ Installation
Clone the Repository

Bash

git clone [https://github.com/sitharthvarsan/Automatic-Number-Plate-Character-Recognition.git](https://github.com/sitharthvarsan/Automatic-Number-Plate-Character-Recognition.git)
cd Automatic-Number-Plate-Character-Recognition
Install Dependencies

Bash

pip install -r requirements.txt
(Manual Install):

Bash

pip install ultralytics paddlepaddle paddleocr opencv-python numpy
🏃‍♂️ Usage
1. Run the Inference Script
This script launches your webcam and waits for the trigger command.

Bash

python scripts/infer_anpr.py
2. Controls
C: Trigger Burst Capture. The system will:

Snap 15 rapid frames.

Detect plates in all frames.

Run Fine-Tuned PaddleOCR on each crop.

Vote for the best text result.

Q: Quit the application.

📊 Training Details
Phase 1: Detection (YOLOv8)
Model: YOLOv8 Nano (yolov8n.pt).

Dataset: Indian License Plates (Roboflow).

Epochs: 50.

Result: High-speed localization of plate coordinates.

Phase 2: Recognition (PaddleOCR)
Architecture: SVTR_LCNet (PP-OCRv3/v4).

Fine-Tuning: Trained on Google Colab using a custom Indian font dataset.

Dictionary: Alphanumeric (0-9, A-Z).

Optimization: Exported to inference model for lightweight deployment.

🤝 Contributing
Contributions are welcome!

Fork the repo.

Create a feature branch (git checkout -b feature/NewFeature).

Commit your changes.

Push to the branch and open a Pull Request.

