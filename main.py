from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
import pickle
import numpy as np
import shutil

MODEL_PATH = "./models/model.pkl"
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

app = FastAPI()

class PredictionRequest(BaseModel):
    inputData: float  # Change this to your actual input data type


@app.post("/predict/")
def predict(request: PredictionRequest):
    try:
        input_data = np.array([[request.inputData]])
        prediction = model.predict(input_data)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "FastAPI server is running and ready to predict!"}


@app.post("/dummy/")
async def handle_file(
    file: UploadFile = File(...),
    extra_param: str = Form(...)
):
    try:
        # Save the uploaded file (optional, for processing or debugging)
        with open(f"temp_{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the file and extra_param (if needed)
        # For example: You might parse the file content or log the data
        print("done")
        return {
            "filename": file.filename,
            "extra_param": extra_param,
            "message": "File and form data received successfully!"
        }
    except Exception as er:
        raise HTTPException(status_code=500, detail=f"Error handling file: {str(er)}")

