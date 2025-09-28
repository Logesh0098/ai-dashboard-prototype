from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.inference import InferenceService

router = APIRouter()
inference_service = InferenceService()

class PredictionInput(BaseModel):
    """Input data model for predictions"""
    features: list[float]

class PredictionOutput(BaseModel):
    """Output data model for predictions"""
    prediction: float
    confidence: float

@router.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        result = inference_service.predict(input_data.features)
        return PredictionOutput(
            prediction=result["prediction"],
            confidence=result["confidence"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))