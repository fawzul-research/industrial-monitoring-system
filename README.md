# industrial-monitoring-system
Industrial Monitoring System: Metal Printing &amp; Alternator Detection

# Industrial Monitoring System: Metal Printing & Alternator Detection

A dual-purpose AI application developed during my internship at the Centre of Excellence (CoE), Universiti Teknikal Malaysia Melaka.  
This system integrates:
- **Real-time metal printing quality monitoring** using image classification (MobileNetV3 + PyTorch)
- **Alternator object detection** using YOLOv8

Built with Python, ONNX, Tkinter, and deployed as a standalone GUI application.

> ⚠️ Note: The training dataset used in this project was collected by my supervisor and team under institutional guidelines. For privacy and research integrity, it is **not publicly shared**. However, the code and model structure are fully available.

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
