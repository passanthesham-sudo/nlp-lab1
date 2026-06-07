"""
FastAPI service exposing sentiment classification and BM25 search.

Endpoints
---------
GET  /health          — liveness + component status
POST /classify        — batch sentiment classification
POST /search          — BM25 search with optional sentiment filter
GET  /experiments     — list MLflow experiments (for debugging)
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.sentiment import SentimentClassifier
from src.preprocessing.pipeline import NLPPipeline
from src.search.engine import BM25SearchEngine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NLP Intelligence API",
    description=(
        "Sentiment classifier + BM25 search engine over Amazon Fine Food Reviews "
        "and Sentiment140. Built for NLP Course Lab 1."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Global state (populated at startup)
# ---------------------------------------------------------------------------

_classifier: Optional[SentimentClassifier] = None
_search_engine: Optional[BM25SearchEngine] = None

MODEL_PATH = os.getenv("MODEL_PATH", "models/sentiment_amazon.pkl")
SEARCH_ENGINE_PATH = os.getenv("SEARCH_ENGINE_PATH", "models/search_engine.pkl")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "")


@app.on_event("startup")
async def _startup() -> None:
    global _classifier, _search_engine

    if MLFLOW_URI:
        mlflow.set_tracking_uri(MLFLOW_URI)

    if Path(MODEL_PATH).exists():
        _classifier = SentimentClassifier.load(MODEL_PATH)
        logger.info("Sentiment classifier loaded.")
    else:
        logger.warning(
            f"No classifier found at '{MODEL_PATH}'. "
            "Run `python experiments/train.py` first."
        )

    if Path(SEARCH_ENGINE_PATH).exists():
        _search_engine = BM25SearchEngine.load(SEARCH_ENGINE_PATH)
        logger.info("Search engine loaded.")
    else:
        logger.warning(
            f"No search engine found at '{SEARCH_ENGINE_PATH}'. "
            "Run `python experiments/train.py` first."
        )


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ClassifyRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="Raw text strings to classify")
    domain: str = Field(
        "reviews",
        description="Preprocessing preset: 'reviews' (Amazon) or 'twitter' (Sentiment140)",
    )


class ClassifyResponse(BaseModel):
    predictions: List[str]
    probabilities: Optional[List[Dict[str, float]]] = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Free-text search query")
    top_k: int = Field(10, ge=1, le=100)
    sentiment_filter: Optional[str] = Field(
        None, description="Filter results: 'positive', 'negative', or null for all"
    )


class SearchResponse(BaseModel):
    results: List[Dict]
    total_found: int
    query: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="Health check")
def health() -> Dict:
    return {
        "status": "ok",
        "classifier_loaded": _classifier is not None,
        "search_engine_loaded": _search_engine is not None,
        "search_index_size": len(_search_engine._documents) if _search_engine else 0,
    }


@app.post("/classify", response_model=ClassifyResponse, summary="Classify sentiment")
def classify(req: ClassifyRequest) -> ClassifyResponse:
    if _classifier is None:
        raise HTTPException(
            503,
            detail="Classifier not loaded. Run `python experiments/train.py` first.",
        )

    config = (
        NLPPipeline.TWITTER_CONFIG
        if req.domain == "twitter"
        else NLPPipeline.REVIEWS_CONFIG
    )
    pipeline = NLPPipeline(config)
    processed = pipeline.batch_process(req.texts)
    predictions = _classifier.predict(processed)

    probabilities: Optional[List[Dict[str, float]]] = None
    try:
        probas = _classifier.predict_proba(processed)
        probabilities = [
            {"negative": round(float(p[0]), 4), "positive": round(float(p[1]), 4)}
            for p in probas
        ]
    except Exception:
        pass

    return ClassifyResponse(predictions=predictions, probabilities=probabilities)


@app.post("/search", response_model=SearchResponse, summary="BM25 document search")
def search(req: SearchRequest) -> SearchResponse:
    if _search_engine is None:
        raise HTTPException(
            503,
            detail="Search engine not loaded. Run `python experiments/train.py` first.",
        )

    if req.sentiment_filter and req.sentiment_filter not in ("positive", "negative"):
        raise HTTPException(
            400, detail="sentiment_filter must be 'positive', 'negative', or null"
        )

    results = _search_engine.search(
        query=req.query,
        top_k=req.top_k,
        sentiment_filter=req.sentiment_filter,
    )
    return SearchResponse(
        results=[r.to_dict() for r in results],
        total_found=len(results),
        query=req.query,
    )


@app.get("/experiments", summary="List MLflow experiments")
def list_experiments() -> List[Dict]:
    try:
        experiments = mlflow.search_experiments()
        return [
            {
                "experiment_id": e.experiment_id,
                "name": e.name,
                "lifecycle_stage": e.lifecycle_stage,
            }
            for e in experiments
        ]
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
