"""
Main training script.

What it does
------------
1. Loads and preprocesses both datasets (Amazon + Sentiment140).
2. Trains every combination of vectorizer × model on each dataset.
3. Logs all runs to MLflow.
4. Saves the best model (highest F1) per dataset to models/.
5. Builds and saves the BM25 search engine from Amazon reviews.

Usage
-----
    python experiments/train.py                        # use default paths
    python experiments/train.py --amazon-sample 20000  # smaller run for testing
    python experiments/train.py --skip-sentiment140    # Amazon only
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.sentiment import SentimentClassifier
from src.preprocessing.pipeline import NLPPipeline
from src.search.engine import BM25SearchEngine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

VECTORIZERS = ["tfidf", "bow"]
MODELS = ["logistic_regression", "linear_svc"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_amazon(path: str, sample: Optional[int] = None) -> Tuple[List[str], List[int]]:
    logger.info(f"Loading Amazon Fine Food Reviews from {path}...")
    df = pd.read_csv(path)
    df = df.dropna(subset=["Text", "Score"])
    df = df[df["Score"] != 3]  # drop neutral 3-star reviews
    df["label"] = (df["Score"] >= 4).astype(int)
    if sample:
        df = df.sample(min(sample, len(df)), random_state=42)
    pos = df["label"].sum()
    logger.info(f"  {len(df):,} reviews  |  {pos:,} positive  {len(df)-pos:,} negative")
    return df["Text"].tolist(), df["label"].tolist()


def load_sentiment140(
    path: str, sample: Optional[int] = 100_000
) -> Tuple[List[str], List[int]]:
    logger.info(f"Loading Sentiment140 from {path}...")
    df = pd.read_csv(
        path,
        encoding="latin-1",
        header=None,
        names=["target", "id", "date", "flag", "user", "text"],
    )
    df["label"] = (df["target"] == 4).astype(int)
    if sample:
        df = df.sample(min(sample, len(df)), random_state=42)
    pos = df["label"].sum()
    logger.info(f"  {len(df):,} tweets  |  {pos:,} positive  {len(df)-pos:,} negative")
    return df["text"].tolist(), df["label"].tolist()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_all_combinations(
    texts: List[str],
    labels: List[int],
    dataset_name: str,
    params: Dict,
) -> Tuple[SentimentClassifier, Dict]:
    """
    Train every vectorizer × model combination.
    Returns the best classifier (by F1) and its metrics.
    """
    max_features = params.get("max_features", 50_000)
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)

    best_clf: Optional[SentimentClassifier] = None
    best_metrics: Optional[Dict] = None

    for vtype in VECTORIZERS:
        for mtype in MODELS:
            clf = SentimentClassifier(
                model_type=mtype,
                vectorizer_type=vtype,
                max_features=max_features,
            )
            metrics = clf.train(
                texts=texts,
                labels=labels,
                test_size=test_size,
                random_state=random_state,
                experiment_name=f"sentiment_{dataset_name}",
                dataset_name=dataset_name,
            )
            if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
                best_metrics = metrics
                best_clf = clf

    return best_clf, best_metrics


def build_search_engine(
    texts: List[str],
    labels: List[int],
    pipeline: NLPPipeline,
    bm25_params: Dict,
) -> BM25SearchEngine:
    label_map = {0: "negative", 1: "positive"}
    sentiments = [label_map[l] for l in labels]

    engine = BM25SearchEngine(
        k1=bm25_params.get("k1", 1.5),
        b=bm25_params.get("b", 0.75),
        pipeline_config=pipeline.config,
    )
    engine.index(texts, sentiments=sentiments)
    return engine


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NLP Lab1 models")
    parser.add_argument("--amazon-path", default="data/raw/Reviews.csv")
    parser.add_argument(
        "--sentiment140-path",
        default="data/raw/training.1600000.processed.noemoticon.csv",
    )
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--amazon-sample", type=int, default=None)
    parser.add_argument("--s140-sample", type=int, default=None)
    parser.add_argument("--skip-sentiment140", action="store_true")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)

    Path("models").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)

    amazon_sample = args.amazon_sample or params["training"]["amazon"].get(
        "sample_size", 50_000
    )
    s140_sample = args.s140_sample or params["training"]["sentiment140"].get(
        "sample_size", 100_000
    )

    # ------------------------------------------------------------------ Amazon
    amazon_path = Path(args.amazon_path)
    if amazon_path.exists():
        texts, labels = load_amazon(str(amazon_path), amazon_sample)
        pipeline = NLPPipeline(NLPPipeline.REVIEWS_CONFIG)

        logger.info("Preprocessing Amazon texts...")
        processed = pipeline.batch_process(texts)

        best_clf, best_metrics = train_all_combinations(
            processed, labels, "amazon", params["training"]["amazon"]
        )
        best_clf.save("models/sentiment_amazon.pkl")
        with open("metrics/amazon_metrics.json", "w") as f:
            json.dump(best_metrics, f, indent=2)
        logger.info(f"Best Amazon model: f1={best_metrics['f1']:.4f}")

        logger.info("Building BM25 search engine on Amazon reviews...")
        engine = build_search_engine(
            texts, labels, pipeline, params["search"]["bm25"]
        )
        engine.save("models/search_engine.pkl")
        logger.info("Search engine saved.")
    else:
        logger.warning(f"Amazon dataset not found at {amazon_path}. Skipping.")

    # ------------------------------------------------------------ Sentiment140
    if not args.skip_sentiment140:
        s140_path = Path(args.sentiment140_path)
        if s140_path.exists():
            texts, labels = load_sentiment140(str(s140_path), s140_sample)
            pipeline = NLPPipeline(NLPPipeline.TWITTER_CONFIG)

            logger.info("Preprocessing Sentiment140 texts...")
            processed = pipeline.batch_process(texts)

            best_clf, best_metrics = train_all_combinations(
                processed, labels, "sentiment140", params["training"]["sentiment140"]
            )
            best_clf.save("models/sentiment_sentiment140.pkl")
            with open("metrics/sentiment140_metrics.json", "w") as f:
                json.dump(best_metrics, f, indent=2)
            logger.info(f"Best Sentiment140 model: f1={best_metrics['f1']:.4f}")
        else:
            logger.warning(f"Sentiment140 dataset not found at {s140_path}. Skipping.")

    logger.info("Training complete. Models saved to models/")
    logger.info("Run `mlflow ui` to explore experiment results.")


if __name__ == "__main__":
    main()
