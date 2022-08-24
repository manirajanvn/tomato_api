from fastapi import FastAPI, File, UploadFile
from app.model.model import predict_disease, read_image
from app.model.model import __version__ as model_version
from fastai.vision.all import *
from json_models.response import Response

app = FastAPI()

@app.post("/predict/", response_model=Response)
async def predict(myfile: UploadFile = File(...)):
    image = read_image(await myfile.read())
    return predict_disease(image)