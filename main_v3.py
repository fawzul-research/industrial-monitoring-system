from model import num2classes, load_model, predict_image
import tkinter as tk
from tkinter import filedialog, Menu
from PIL import Image, ImageTk
import cv2
import numpy as np
import albumentations as A

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from yolov8 import YOLOv8

# Initialize yolov8 object detector
model_path = "best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.3)


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
        self.window.resizable(False, False)
        # Set fixed window size
        self.window_width = 1100
        self.window_height = 800

        self.stats = {'bad': 1, 'good': 1}
        
        self.menubar = Menu(self.window)
        self.filemenu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.filemenu)
        self.filemenu.add_command(label="Open", command=self.donothing)
        self.filemenu.add_command(label="Save", command=self.donothing)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self.window.quit)
        
        self.editmenu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Edit", menu=self.editmenu)
        self.editmenu.add_command(label="Element 1", command=self.donothing)
        self.editmenu.add_separator()
        self.editmenu.add_command(label="Element 2", command=self.donothing)

        self.helpmenu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=self.helpmenu)
        self.helpmenu.add_command(label="Element 1", command=self.donothing)
        self.helpmenu.add_command(label="Element 2", command=self.donothing)


        self.model_path = model_path
        self.ort_sess = load_model(self.model_path)

        self.cap = None
        self.video_path = None
        self.stop_camera = False  # New instance variable
        
    def donothing(self):
        if (self.var1.get() == 0 and self.var2.get() == 0) or (self.var1.get() == 1 and self.var2.get() == 1):
            if self.checked == 1:
                self.var1.set(0)
                self.var2.set(1)
                self.checked = 2
            else:
                self.var1.set(1)
                self.var2.set(0)
                self.checked = 1
        print('metal: ', self.var1.get())
        print('defect: ', self.var2.get())
    
    def draw_pie_chart(self):
        # Data for the pie chart
        labels = list(self.stats.keys())
        sizes = list(self.stats.values())

        # Calculate the size in inches based on pixels and DPI
        dpi = plt.rcParams['figure.dpi']
        width_px = 500  # Specify the width in pixels
        height_px = 400  # Specify the height in pixels
        width_inch = width_px / dpi
        height_inch = height_px / dpi

        # Create a figure with desired size (in inches)
        self.fig, self.ax = plt.subplots(figsize=(width_inch, height_inch))

        # Draw the pie chart
        self.ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        self.ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

        # Display the pie chart in the Tkinter GUI
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas_plot.draw()
        self.canvas_plot.get_tk_widget().place(x=0, y=440)
        # self.canvas_plot.place(x=10,y=10)
    
    def update_pie_chart(self):
        # Data for the pie chart
        labels = list(self.stats.keys())
        sizes = list(self.stats.values())

        # Clear previous plot
        self.ax.clear()
        
        # Draw the pie chart
        self.ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        self.ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

        # Display the pie chart in the Tkinter GUI
        self.canvas_plot.draw()

    
    def display_image(self, file_path):
        try:
            if self.var1.get() == 1 and self.var2.get() == 0:
                prediction = predict_image(file_path, self.ort_sess)
                self.result_label.config(text=f"Prediction: {num2classes[prediction]}", font=("Helvetica", 16))
                image = Image.open(file_path)
            else:
                img = cv2.imread(file_path)
                boxes, scores, class_ids = yolov8_detector(img)
                # Draw detections
                combined_img = yolov8_detector.draw_detections(img)
                image = Image.fromarray(combined_img)
            image.thumbnail((500, 400))
            photo = ImageTk.PhotoImage(image)
            self.canvas.config(width=image.width, height=image.height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo
        except Exception as e:
            self.result_label.config(text=f"Error loading image: {e}", font=("Helvetica", 16))

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Files", ".png .jpg .jpeg .avi .mp4")])
        file_ext = file_path.split(".")[-1].lower()
        if file_ext in ['avi', 'mp4']:
            self.stop_camera = False
            self.cap = cv2.VideoCapture(file_path)
            self.update_camera()
        if file_ext in ['jpg', 'png', 'jpeg']:
            self.display_image(file_path)
        pass
    
    def open_cam(self):
        self.stop_camera = False
        self.video_path = 0
        self.cap = cv2.VideoCapture(self.video_path)
        self.update_camera()

    def release_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def stop_camera_video(self):
        self.stop_camera = True
        self.release_camera()
        self.stats = {'bad': 1, 'good': 1}

    def update_camera(self):
        if self.cap is None:  # Check if camera is released
            print("Camera/Video stopped.")
            return

        ret, frame = self.cap.read()

        if not ret:
            print("End of video. Exiting.")
            self.release_camera()
            return
        if self.var1.get() == 1 and self.var2.get() == 0:
            prediction = predict_frame(frame, self.ort_sess)
            
            if num2classes[prediction] == "bad":
                self.stats['bad'] += 1
            else:
                self.stats['good'] += 1
            
            self.update_pie_chart()
            self.result_label.config(text=f"Prediction: {num2classes[prediction]}", font=("Helvetica", 16))
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            boxes, scores, class_ids = yolov8_detector(frame)
            # Draw detections
            combined_img = yolov8_detector.draw_detections(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = Image.fromarray(combined_img)
        
        image.thumbnail((500, 400))
        photo = ImageTk.PhotoImage(image)
        self.canvas.config(width=image.width, height=image.height)
        # print(image.width, image.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo
        self.canvas.place(x=550,y=50)


        self.window.after(10, self.update_camera)

    def run(self):
        self.window.geometry(f"{self.window_width}x{self.window_height}")
        
        self.draw_pie_chart()
        
        self.canvas = tk.Canvas(self.window, width=500, height=400, bg='black')
        self.canvas.pack()
        self.canvas.place(x=550,y=50)
        
        self.result_label = tk.Label(self.window, text="Select an image to classify.", font=("Helvetica", 16))
        self.result_label.pack(pady=10)
        self.result_label.place(x=550,y=10)
        
        self.open_cam_btn = tk.Button(self.window, text="Open Cam", command=self.open_cam, font=("Helvetica", 14))
        self.open_cam_btn.pack(pady=10)
        self.open_cam_btn.place(x=10,y=120)
        
        self.open_file_btn = tk.Button(self.window, text="Open image/video", command=self.open_file, font=("Helvetica", 14))
        self.open_file_btn.pack(pady=10)
        self.open_file_btn.place(x=10,y=159)
        
        self.stop_camera_btn = tk.Button(self.window, text="Stop Camera/Video", command=self.stop_camera_video,
                                    font=("Helvetica", 14))
        self.stop_camera_btn.pack(pady=10)
        self.stop_camera_btn.place(x=10,y=80)
        
        # self.plot_btn = tk.Button(self.window, text="plot", command=self.draw_pie_chart,
        #                             font=("Helvetica", 14))
        # self.plot_btn.pack(pady=10)
        # self.plot_btn.place(x=10,y=120)
        
        # exit_button = tk.Button(self.window, text="Exit", command=self.window.destroy, font=("Helvetica", 14),
        #                         bg="red", fg="white")
        # exit_button.pack(pady=10)

        # self.selected_option = tk.StringVar()
        # self.selected_option.set("Metal printing monitoring.")  # Set default option

        # options = ["Metal printing monitoring.", "Defect detection         ."]

        # self.option_menu = tk.OptionMenu(self.window, self.selected_option, *options)
        # self.option_menu.pack()
        # self.option_menu.place(x=10, y=10)
        
        self.checked = 1
        self.var1 = tk.IntVar(value=1)
        self.var2 = tk.IntVar()
        self.c1 = tk.Checkbutton(self.window, text='Metal printing monitoring',variable=self.var1, font=("Helvetica", 14),
                            onvalue=1, offvalue=0, command=self.donothing)
        self.c1.pack()
        self.c1.place(x=10,y=10)
        
        self.c2 = tk.Checkbutton(self.window, text='Defect detection',variable=self.var2, font=("Helvetica", 14),
                            onvalue=1, offvalue=0, command=self.donothing)
        self.c2.pack()
        self.c2.place(x=10,y=40)
        self.window.config(menu=self.menubar)
        self.window.mainloop()

def main():
    model_path = 'mobilenet.onnx'
    root = tk.Tk()
    app = CameraApp(root, model_path)
    app.run()

if __name__ == "__main__":
    main()