from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os

app = FastAPI()

class InputData(BaseModel):
    input_value: float

MODEL_PATH = "./models/model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"The specified model file '{MODEL_PATH}' does not exist.")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI with a .pkl model!"}

@app.post("/predict")
def predict(data: InputData):
    try:
        # Extract input and make a prediction
        input_value = data.input_value
        result = model.predict([[input_value]])
        return {"input": input_value, "prediction": result[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")