"""
FastAPI app that exposes /predict.
POST /predict with JSON: {"inputs": [[...], [...]]}
Returns: {"predictions": [probabilities]}
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import src.model as model_module

app = FastAPI(title="Simple MLOps Demo API")

class PredictRequest(BaseModel):
    inputs: List[List[float]]

class PredictResponse(BaseModel):
    predictions: List[float]

@app.get("/")
def root():
    return {"status": "ok", "note": "POST /predict with inputs"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        preds = model_module.predict(req.inputs)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
    return {"predictions": preds}