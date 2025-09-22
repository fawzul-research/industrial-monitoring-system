import onnxruntime as ort
import numpy as np
from PIL import Image
import albumentations as A

num2classes = {0:'bad', 1:'good'}

transforms = A.Compose([
        A.Resize(265, 265),
        A.CenterCrop(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_model(model_path):
    return ort.InferenceSession(model_path)

def preprocess_image(image_path, transforms):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    transformed_image = transforms(image=image_np)["image"].transpose(2, 0, 1)
    sample = np.expand_dims(transformed_image, axis=0)
    return sample

def predict_image(image_path, model, threshold=0.5):
    outputs = model.run(None, {'actual_input': preprocess_image(image_path, transforms)})
    prediction = 0 if sigmoid(outputs[0][0][0]) < threshold else 1
    return prediction
