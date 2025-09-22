from model import num2classes, load_model, predict_image

import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

import cv2
import numpy as np

import albumentations as A

transforms = A.Compose([
    A.Resize(265, 265),
    A.CenterCrop(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess_image(frame, transforms):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_np = np.array(image)
    transformed_image = transforms(image=image_np)["image"].transpose(2, 0, 1)
    sample = np.expand_dims(transformed_image, axis=0)
    return sample


def predict_frame(frame, model):
    outputs = model.run(None, {'actual_input': preprocess_image(frame, transforms)})
    prediction = 0 if sigmoid(outputs[0][0][0]) < 0.5 else 1
    return prediction


class CameraApp:
    def __init__(self, window, model_path):
        self.window = window
        self.window.title("Image Classifier")

        self.canvas = tk.Canvas(self.window, width=400, height=400)
        self.canvas.pack()

        self.result_label = tk.Label(self.window, text="Select an image to classify.", font=("Helvetica", 16))
        self.result_label.pack(pady=10)

        self.model_path = model_path
        self.ort_sess = load_model(self.model_path)

        self.cap = None
        self.video_path = None

    def display_image(self, file_path):
        try:
            image = Image.open(file_path)
            image.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(image)
            self.canvas.config(width=image.width, height=image.height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo
            prediction = predict_image(file_path, self.ort_sess)
            self.result_label.config(text=f"Prediction: {num2classes[prediction]}", font=("Helvetica", 16))
        except Exception as e:
            self.result_label.config(text=f"Error loading image: {e}", font=("Helvetica", 16))

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Files", ".png .jpg .jpeg .avi .mp4")])
        file_ext = file_path.split(".")[-1].lower()
        if file_ext in ['avi']:
            self.cap = cv2.VideoCapture(file_path)
            self.update_camera()
        if file_ext in ['jpg', 'png', 'jpeg']:
            self.display_image(file_path)
        pass
    
    def open_cam(self):
        self.video_path = 0
        self.cap = cv2.VideoCapture(self.video_path)
        self.update_camera()

    def update_camera(self):
        ret, frame = self.cap.read()

        if not ret:
            print("End of video. Exiting.")
            return

        prediction = predict_frame(frame, self.ort_sess)
        cv2.putText(frame, f"Prediction: {num2classes[prediction]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(image)
        self.canvas.config(width=image.width, height=image.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

        self.result_label.config(text=f"Prediction: {num2classes[prediction]}", font=("Helvetica", 16))

        self.window.after(10, self.update_camera)

    def run(self):
        open_cam_btn = tk.Button(self.window, text="Open Cam", command=self.open_cam, font=("Helvetica", 14))
        open_cam_btn.pack(pady=10)
        
        open_file_btn = tk.Button(self.window, text="Open image/video", command=self.open_file, font=("Helvetica", 14))
        open_file_btn.pack(pady=10)

        exit_button = tk.Button(self.window, text="Exit", command=self.window.destroy, font=("Helvetica", 14),
                                bg="red", fg="white")
        exit_button.pack(pady=10)
        
        self.window.mainloop()


def main():
    model_path = 'mobilenet.onnx'

    root = tk.Tk()
    app = CameraApp(root, model_path)
    app.run()


if __name__ == "__main__":
    main()