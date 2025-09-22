# Industrial Monitoring System: 3D Metal Printing & Alternator Detection
---

### Project Overview

This project is an **AI-powered Industrial Monitoring System** that I developed during my internship at Universiti Teknikal Malaysia Melaka. It focuses on improving quality control and safety in factories by using computer vision.

The system has two main functions:

**(i) Metal Printing Quality Monitoring** – Uses a deep learning model (MobileNetV3 with PyTorch) to check printed metal parts. It works in two ways:

* **Live (Real-Time) Monitoring** – Inspects parts during production and instantly flags defects or irregularities.

* **Post-Check Mode** – Users can upload images or videos after production to analyze quality.

Together, these options help reduce waste, maintain standards, and give flexibility to factory operators.

**(ii) Alternator Detection** – Applies object detection (YOLOv8) to automatically identify alternators from live camera feeds. This can be used on assembly lines for faster, more reliable inspections.

<br>

## Why this is useful:

* Saves time by automating inspections that were previously manual.

* Reduces errors and increases production accuracy.

* Provides a simple interface so operators can use it without deep technical knowledge.

In short, this system shows how AI can be applied in real factories to make production smarter, faster, and more reliable.

<br>

## Features
* **Real-Time Camera Monitoring**<br>
Capture and process live video streams from cameras to check product quality instantly.

* **Automatic Defect Detection**<br>
Detects issues in printed metal parts, reducing human error and improving accuracy.

* **Alternator Identification**<br>
Uses AI to recognize alternators on the production line, helping with faster inspection.

* **Image/Video Testing Mode**<br>
You can also upload saved images/videos (instead of using a live camera) to run quality checks.

* **User-Friendly Interface**<br>
Simple design that allows operators to run the system without deep technical skills.

* **Customizable Models**<br>
Built with ONNX and YOLOv8 so models can be retrained or replaced for other factory needs.

<br>

## Technologies Used
- Python, PyTorch, ONNX, OpenCV
- YOLOv8 (Ultralytics), MobileNetV3
- Tkinter (GUI), Flask (optional backend)
- Google Colab (training), PyInstaller (deployment)

> Note: The training dataset used in this project was collected by my supervisor and team under institutional guidelines. For privacy and research integrity, it is **not publicly shared**. However, the code and model structure are fully available.
 
<br>

## Project Structure
```
industrial-monitoring-system/
├── main_v3.py              # Main GUI application
├── camera_inside.py        # simple OpenCV + Tkinter camera feed
├── test_object_detection.py # Test script
├── model.py                # Model loading logic
├── requirements.txt        # Dependencies
├── best.onnx               # YOLOv8 model (ONNX format)
├── mobilenet.onnx          # Image classifier model
├── Plot_only.py            # pie chart in Tkinter
├── full_with_cam_inside.py #complete GUI app with model loading, file/camera handling, prediction
└── images_for_test/        # Sample test images (anonymized)
```
<br>

## How to Run

### 🐧 Linux (Ubuntu/Debian)

**1️⃣ Install Git (to download the project)**
```bash

sudo apt update
sudo apt install git -y
```
**2️⃣ Install Python (if not already installed)**
<br>Check version:
```bash

python3 --version
```
If not installed:<br>
```bash

sudo apt install python3 python3-venv python3-pip -y
```
**3️⃣ Download the project**<br>
```bash

git clone https://github.com/fawzul-research/industrial-monitoring-system.git
cd industrial-monitoring-system
```
**4️⃣ Create a virtual environment**<br>
```bash
python3 -m venv venv
source venv/bin/activate
```
**5️⃣ Install dependencies**<br>
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
**6️⃣ Run the program**<br>
```bash
python main_v3.py
```
**7️⃣ Exit the environment when finished**<br>
```bash
deactivate
```
<br>

### 🪟 Windows
**1️⃣ Install Git for Windows**<br>
```bash
Download from: https://git-scm.com/download/win
```
During installation, keep default settings.

**2️⃣ Install Python**<br>
Download from: ```https://www.python.org/downloads/windows/ ```
<br>During installation:<br>
✅ Check “Add Python to PATH”<br>
✅ Keep default options

**3️⃣ Open Git Bash or Command Prompt and download the project:**<br>
```bash
git clone https://github.com/fawzul-research/industrial-monitoring-system.git
cd industrial-monitoring-system
```
**4️⃣ Create a virtual environment**<br>
```bash
python -m venv venv
venv\Scripts\activate
```
**5️⃣ Install dependencies**<br>
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
**6️⃣ Run the program**<br>
```bash
python main_v3.py
```
**7️⃣ Exit the environment when finished**<br>
```bash
venv\Scripts\deactivate
```

