# industrial-monitoring-system
Industrial Monitoring System: Metal Printing &amp; Alternator Detection

# Industrial Monitoring System: Metal Printing & Alternator Detection

A dual-purpose AI application developed during my internship at the Centre of Excellence (CoE), Universiti Teknikal Malaysia Melaka.  
This system integrates:
- **Real-time metal printing quality monitoring** using image classification (MobileNetV3 + PyTorch)
- **Alternator object detection** using YOLOv8

Built with Python, ONNX, Tkinter, and deployed as a standalone GUI application.

> Note: The training dataset used in this project was collected by my supervisor and team under institutional guidelines. For privacy and research integrity, it is **not publicly shared**. However, the code and model structure are fully available.

---

## Features
- Detects defects in metal prints in real time
- Identifies industrial alternators from live camera feeds
- Integrates both models into a single GUI app using Tkinter
- Optimized for production use with ONNX model conversion
- Deployed as standalone executable via PyInstaller

---

## Technologies Used
- Python, PyTorch, ONNX, OpenCV
- YOLOv8 (Ultralytics), MobileNetV3
- Tkinter (GUI), Flask (optional backend)
- Google Colab (training), PyInstaller (deployment)

---

## Project Structure
```
industrial-monitoring-system/
├── main_v3.py              # Main GUI application
├── camera_inside.py        # Camera feed handler
├── test_object_detection.py # Test script
├── model.py                # Model loading logic
├── requirements.txt        # Dependencies
├── best.onnx               # YOLOv8 model (ONNX format)
├── mobilenet.onnx          # Image classifier model
├── output.mp4              # Demo video of system in action
├── detected_objects.jpg    # Sample detection result
└── images_for_test/        # Sample test images (anonymized)
```
---

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
---

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

