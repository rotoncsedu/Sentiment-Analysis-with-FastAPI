
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predictor import SentimentPredictor

app = FastAPI(
    title="Sentiment Analysis Inference API",
    description="Serves a GRU model trained on the IMDB movie review dataset.",
    version="1.0.0",
)

# Load the predictor once at startup, not inside the predict function.
predictor = SentimentPredictor()


class ReviewInput(BaseModel):
    text: str = Field(..., min_length=1, description="Movie review text")


class SentimentOutput(BaseModel):
    label: str          # "POSITIVE" or "NEGATIVE"
    probability: float


@app.get("/")
def root():
    return {
        "name": "Sentiment Analysis Inference API",
        "description": (
            "A REST API serving a GRU-based sentiment classifier "
            "trained on the IMDB movie review dataset. "
            "POST a review to /predict to get a POSITIVE/NEGATIVE "
            "label with a probability score."
        ),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=SentimentOutput)
def predict(review: ReviewInput):
    try:
        result = predictor.predict(review.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return SentimentOutput(**result)
