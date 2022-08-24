from fastapi import FastAPI, File, UploadFile
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version


app = FastAPI()


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}



@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    result = predict_pipeline(await file.read())
    return result
