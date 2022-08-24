from typing import Dict
from io import BytesIO
import numpy as np
from PIL import Image
from fastai.vision.all import *

__version__ = "1"

classes = [
    "Bacterial spot",
    "Early blight",
    "Late blight",
    "Leaf Mold",
    "Septoria leaf spot",
    "Spider mites Two-spotted spider mite",
    "Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato mosaic virus",
    "healthy"
]

def read_image(file: bytes) -> PILImage:
    img = Image.open(BytesIO(file))
    fastimg = PILImage.create(np.array(img.convert('RGB')))

    return fastimg

def is_hotdog(x):
    return "frankfurter" in x or "hotdog" in x or "chili-dog" in x

def predict_disease(image) -> Dict:
    path = Path()
    inference_model = load_learner(path/'model.pkl')
    prediction = inference_model.predict(image)
    return classes[prediction[0]]