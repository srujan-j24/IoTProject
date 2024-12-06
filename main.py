from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

MODEL_PATH = "./models/model.pkl"
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

app = FastAPI()

class PredictionRequest(BaseModel):
    feature: float

@app.post("/predict/")
def predict(request: PredictionRequest):
    try:
        feature = np.array([[request.feature]])
        prediction = model.predict(feature)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "FastAPI server is running and ready to predict!"}
