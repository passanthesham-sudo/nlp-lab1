"""
Sentiment classifier with MLflow experiment tracking.

Supports multiple vectorizer + model combinations so they can be compared
in the same MLflow experiment and the best config promoted to production.
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.vectorization.bow import BagOfWords
from src.vectorization.tfidf import TFIDF

logger = logging.getLogger(__name__)

_MODELS = {
    "logistic_regression": lambda: LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs"
    ),
    "linear_svc": lambda: LinearSVC(C=1.0, max_iter=2000),
    "naive_bayes": lambda: MultinomialNB(alpha=0.1),
}

_VECTORIZERS = {
    "tfidf": lambda mf: TFIDF(max_features=mf),
    "bow": lambda mf: BagOfWords(max_features=mf),
}

_LABEL_MAP = {0: "negative", 1: "positive"}


class SentimentClassifier:
    """
    Train, evaluate, save, and load a sentiment classifier.

    Usage:
        clf = SentimentClassifier(model_type="logistic_regression", vectorizer_type="tfidf")
        metrics = clf.train(processed_texts, labels, experiment_name="amazon")
        predictions = clf.predict(new_processed_texts)
        clf.save("models/sentiment_amazon.pkl")
    """

    def __init__(
        self,
        model_type: str = "logistic_regression",
        vectorizer_type: str = "tfidf",
        max_features: int = 50000,
    ) -> None:
        if model_type not in _MODELS:
            raise ValueError(f"model_type must be one of {list(_MODELS)}")
        if vectorizer_type not in _VECTORIZERS:
            raise ValueError(f"vectorizer_type must be one of {list(_VECTORIZERS)}")

        self.model_type = model_type
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.vectorizer = _VECTORIZERS[vectorizer_type](max_features)
        self.model = _MODELS[model_type]()
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        texts: List[str],
        labels: List[int],
        test_size: float = 0.2,
        random_state: int = 42,
        experiment_name: str = "sentiment_classification",
        run_name: Optional[str] = None,
        dataset_name: str = "unknown",
    ) -> Dict[str, float]:
        """Train and evaluate; log everything to MLflow. Returns test metrics."""
        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        mlflow.set_experiment(experiment_name)
        rname = run_name or f"{dataset_name}__{self.vectorizer_type}__{self.model_type}"

        with mlflow.start_run(run_name=rname):
            mlflow.log_params(
                {
                    "model_type": self.model_type,
                    "vectorizer_type": self.vectorizer_type,
                    "max_features": self.max_features,
                    "test_size": test_size,
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "dataset": dataset_name,
                }
            )

            logger.info(f"Vectorizing with {self.vectorizer_type}...")
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)

            logger.info(f"Training {self.model_type}...")
            self.model.fit(X_train_vec, y_train)
            self.is_fitted = True

            y_pred = self.model.predict(X_test_vec)
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred, average="weighted")),
                "precision": float(
                    precision_score(y_test, y_pred, average="weighted")
                ),
                "recall": float(recall_score(y_test, y_pred, average="weighted")),
            }

            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(self.model, "model")

            logger.info(
                f"[{dataset_name}] {self.vectorizer_type}+{self.model_type} → "
                f"acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}"
            )
            logger.info(
                "\n"
                + classification_report(
                    y_test, y_pred, target_names=["negative", "positive"]
                )
            )

        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, texts: List[str]) -> List[str]:
        """Expects preprocessed (pipeline-processed) strings."""
        self._check_fitted()
        X = self.vectorizer.transform(texts)
        return [_LABEL_MAP[int(p)] for p in self.model.predict(X)]

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        self._check_fitted()
        X = self.vectorizer.transform(texts)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        # LinearSVC: sigmoid-calibrated decision function
        scores = self.model.decision_function(X)
        if scores.ndim == 1:
            prob = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack([1.0 - prob, prob])
        return scores

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "vectorizer": self.vectorizer,
            "model_type": self.model_type,
            "vectorizer_type": self.vectorizer_type,
            "max_features": self.max_features,
            "is_fitted": self.is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"Saved model to {path}")

    @classmethod
    def load(cls, path: str) -> "SentimentClassifier":
        with open(path, "rb") as f:
            data = pickle.load(f)
        inst = cls(
            model_type=data["model_type"],
            vectorizer_type=data["vectorizer_type"],
            max_features=data.get("max_features", 50000),
        )
        inst.model = data["model"]
        inst.vectorizer = data["vectorizer"]
        inst.is_fitted = data["is_fitted"]
        logger.info(f"Loaded model from {path}")
        return inst

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model is not trained. Call train() or load() first.")
