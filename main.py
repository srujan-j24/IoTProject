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
    inputData: float # you should change this to your input data type

@app.post("/predict/")
def predict(request: PredictionRequest):
    try:
        input_data= np.array([[request.inputData]])
        prediction = model.predict(input_data)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "FastAPI server is running and ready to predict!"}
