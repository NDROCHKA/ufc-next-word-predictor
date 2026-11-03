"""FastAPI service consumed only by the Express real-time gateway."""

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ufc_next_word.config import ARTIFACTS_DIR
from ufc_next_word.inference.predictor import NextWordPredictor


app = FastAPI(
    title="UFC Next Word Model Service",
    version="1.0.0",
    description="Inference for the assignment's BiRNN and TextCNN models.",
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


@lru_cache(maxsize=1)
def get_predictor() -> NextWordPredictor:
    """Load model artifacts once per Python process, not once per keystroke."""

    return NextWordPredictor(ARTIFACTS_DIR)


@app.get("/health")
def health() -> dict:
    required = ("birnn.pt", "textcnn.pt", "vocabulary.json", "model_config.json")
    missing = [name for name in required if not (ARTIFACTS_DIR / name).exists()]
    return {"status": "ready" if not missing else "models_missing", "missing": missing}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    try:
        return get_predictor().predict(request.text)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are missing. Run data preparation and training first.",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

