import logging
import pandas as pd
import os
from fastapi import APIRouter

from trial_conversion_model.api.schemas import PredictionRequest, PredictionResponse
from trial_conversion_model.predict import load_model, predict_proba
from importlib.metadata import version

logger = logging.getLogger(__name__)

router = APIRouter()

# Load the model once, when the service starts, not on every request.
model = load_model()


def to_band(probability: float) -> str:
    """Turn a raw probability into a label a human can act on."""
    if probability < 0.33:
        return "low"
    if probability < 0.66:
        return "medium"
    return "high"


@router.get("/health")
def health() -> dict:
    return {"status": "ok",
            "version":version("trial-conversion-model"),
            "git_sha":os.environ.get("GIT_SHA", "unknown")
            }


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict conversion probability for a single live trial."""
    # TODO: three steps, roughly one line each:
    #   1. Turn the request into a one-row DataFrame (model_dump gives you a dict).
    #   2. Score it with predict_proba and round to 4 decimals.
    #   3. Turn the probability into a band, and return the PredictionResponse.
    row = pd.DataFrame([request.model_dump()])
    probability = round(float(predict_proba(model, row).iloc[0]), 4)
    return PredictionResponse(
        conversion_probability=probability, 
        conversion_band=to_band(probability),
        )
