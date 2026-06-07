"""
Empirically compare BoW, TF-IDF, and BM25 on the Amazon dataset.

This script produces a side-by-side metrics table and logs every run to MLflow
under the experiment 'vectorizer_comparison'.

Key insight it demonstrates
---------------------------
- TF-IDF outperforms BoW because high-frequency generic words (e.g., "product",
  "item") get down-weighted, leaving the classifier to focus on discriminative terms.
- BM25 excels at retrieval but is not a native classifier; its "pseudo accuracy"
  is computed by scoring documents against seed queries.

Usage
-----
    python experiments/compare_vectorizers.py
    python experiments/compare_vectorizers.py --sample 5000
"""
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.sentiment import SentimentClassifier
from src.preprocessing.pipeline import NLPPipeline
from src.vectorization.bm25 import BM25Vectorizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BM25 pseudo-classifier
# ---------------------------------------------------------------------------


def bm25_pseudo_accuracy(
    texts: List[str], labels: List[int], pipeline: NLPPipeline
) -> float:
    """
    Use BM25 to pseudo-classify by comparing relevance to positive vs negative
    seed queries. Demonstrates BM25 as a retrieval tool, not a classifier.
    """
    pos_seeds = pipeline.process("great excellent amazing love perfect delicious")
    neg_seeds = pipeline.process("terrible awful disgusting horrible bad disappointed")

    tokenized = [pipeline.process(t) for t in texts]
    bm25 = BM25Vectorizer()
    bm25.fit(tokenized)

    pos_scores = bm25.get_scores(pos_seeds)
    neg_scores = bm25.get_scores(neg_seeds)

    predicted = (pos_scores > neg_scores).astype(int)
    return float((predicted == np.array(labels)).mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--amazon-path", default="data/raw/Reviews.csv")
    parser.add_argument("--sample", type=int, default=10_000)
    args = parser.parse_args()

    amazon_path = Path(args.amazon_path)
    if not amazon_path.exists():
        logger.error(f"Dataset not found: {amazon_path}")
        logger.error("Download Reviews.csv from Kaggle and place it in data/raw/")
        return

    df = pd.read_csv(str(amazon_path)).dropna(subset=["Text", "Score"])
    df = df[df["Score"] != 3]
    df["label"] = (df["Score"] >= 4).astype(int)
    df = df.sample(min(args.sample, len(df)), random_state=42)

    raw_texts = df["Text"].tolist()
    labels = df["label"].tolist()
    pipeline = NLPPipeline(NLPPipeline.REVIEWS_CONFIG)
    processed = pipeline.batch_process(raw_texts)

    mlflow.set_experiment("vectorizer_comparison")
    results = {}

    for vtype in ["tfidf", "bow"]:
        clf = SentimentClassifier(
            model_type="logistic_regression",
            vectorizer_type=vtype,
            max_features=50_000,
        )
        metrics = clf.train(
            processed,
            labels,
            experiment_name="vectorizer_comparison",
            run_name=f"{vtype}__logistic_regression",
            dataset_name="amazon_sample",
        )
        results[f"{vtype}_lr"] = metrics

    # BM25 pseudo-accuracy (logged separately)
    bm25_acc = bm25_pseudo_accuracy(raw_texts, labels, pipeline)
    with mlflow.start_run(run_name="bm25_pseudo_classifier"):
        mlflow.log_metric("bm25_pseudo_accuracy", bm25_acc)
    results["bm25_pseudo"] = {"accuracy": bm25_acc, "f1": "N/A (retrieval metric)"}

    # Print summary table
    print("\n" + "=" * 60)
    print("VECTORIZER COMPARISON RESULTS")
    print("=" * 60)
    for name, m in results.items():
        acc = m.get("accuracy", "—")
        f1 = m.get("f1", "—")
        acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
        f1_str = f"{f1:.4f}" if isinstance(f1, float) else str(f1)
        print(f"  {name:<25}  accuracy={acc_str}  f1={f1_str}")
    print("=" * 60)
    print("View full run details: mlflow ui")


if __name__ == "__main__":
    main()
