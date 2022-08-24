import pickle
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms

__version__ = "1"

BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASE_DIR}/tomato_disease_1.pkl", "rb") as f:
    model = pickle.load(f)


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


def predict_pipeline(image_file):
    image_bytes = image_file.file.read()
    class_id,class_name = get_prediction(image_bytes)
    result = {
        "predictions":{
            "class_id":class_id,
            "class_name":class_name
        }
    }
    return result

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return classes[predicted_idx]