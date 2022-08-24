from fastapi import FastAPI, File, UploadFile
from app.model.model import predict_disease, read_image
from app.model.model import __version__ as model_version
from fastai.vision.all import *

app = FastAPI()

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict/")
async def predict(myfile: UploadFile = File(...)):
    image = read_image(await myfile.read())
    return predict_disease(image)