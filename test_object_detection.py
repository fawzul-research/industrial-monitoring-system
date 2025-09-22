import cv2
from yolov8 import YOLOv8

# Initialize yolov8 object detector
model_path = "best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.3)

# Read image
img = cv2.imread('Good Alternator_99.jpg')

# Detect Objects
boxes, scores, class_ids = yolov8_detector(img)

# Draw detections
combined_img = yolov8_detector.draw_detections(img)
# cv2.imwrite("detected_objects.jpg", combined_img)

