# 🚘 Automatic Number Plate Recognition (ANPR) with YOLOv5  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)  
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C?logo=pytorch&logoColor=white)  
![YOLOv5](https://img.shields.io/badge/YOLOv5-Object%20Detection-green)  
![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)  

An end-to-end **Automatic Number Plate Recognition (ANPR)** system built using **YOLOv5**.  
It can detect vehicle license plates from **images, videos, or live streams**, crop them, and is extendable with OCR for plate text recognition.  

---

## ✨ Features

✅ Real-time license plate detection  
✅ Works with **images, videos, and live camera streams**  
✅ Saves cropped license plate images for further processing  
✅ Multiple model support (**YOLOv5 / YOLOv8 / MobileNet**)  
✅ Easy to extend with OCR (Tesseract / CRNN / EasyOCR)  

---

---

## ⚙️ Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/BVISHNU78/ANPR-Using-YOLOV5.git
   cd ANPR-Using-YOLOV5
ANPR-Using-YOLOv5/
├── saved_plates/ # Cropped detected license plates
├── templates/ # Templates/configs (if used)
├── yolov5nu.pt # YOLOv5 pretrained weights
├── yolv8m.pt # YOLOv8 (medium) weights
├── yolv8n.pt # YOLOv8 (nano) weights
├── mobilenetv2_bottleneck_wts.pt# MobileNetV2 weights
├── stream.py # Main script (camera/video detection)
├── requirements.txt # Dependencies
├── cameras.json # Camera source configs
├── lines.json # Config for guiding detection lines
├── anpr.log # Detection logs
├── LICENSE # License file
└── README.md # Project documentation


---

## ⚙️ Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/BVISHNU78/ANPR-Using-YOLOV5.git
   cd ANPR-Using-YOLOV5


Create virtual environment (optional)

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate


Install dependencies

pip install -r requirements.txt


Download / verify weights
Ensure *.pt weight files are in the root folder.

🚀 Usage
▶️ Run on Webcam / Camera
python stream.py --source 0

▶️ Run on Video File
python stream.py --source path/to/video.mp4

▶️ Run on Images
python stream.py --source path/to/image.jpg


➡️ Detected plates will be highlighted in the output and saved in saved_plates/.

📊 Results
Metric	Value (example)
mAP@0.5	0.85
Precision	0.88
Recall	0.82
Inference FPS	~20 (GPU)

## 🚀 Usage

### ▶️ Run on Webcam / Camera
```bash
python stream.py --source 0
▶️ Run on Video File
python stream.py --source path/to/video.mp4

▶️ Run on Images
python stream.py --source path/to/image.jpg
## 📂 Project Structure

▶️ Run on RTSP Stream
python stream.py --source "rtsp://username:password@IP:PORT/stream"


➡️ Example:
python stream.py --source "rtsp://admin:1234@192.168.1.10:554/live"
