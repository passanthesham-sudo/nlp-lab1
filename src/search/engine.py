"""
BM25 search engine with sentiment filtering.

Designed to scale to 500K+ documents. The sentiment filter is a post-retrieval
step: BM25 ranks all documents, then we keep only those matching the requested
sentiment label before slicing top-k.
"""
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.preprocessing.pipeline import NLPPipeline, PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    doc_id: int
    text: str
    score: float
    sentiment: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "text": self.text[:500],
            "score": round(self.score, 4),
            "sentiment": self.sentiment,
            "metadata": self.metadata,
        }


class BM25SearchEngine:
    """
    BM25-based document retrieval engine with optional sentiment filtering.

    Index once; query many times. Serializes to disk so the API can load
    a pre-built index at startup without re-indexing.

    Example::
        engine = BM25SearchEngine()
        engine.index(texts, sentiments=labels)
        results = engine.search("great taste", top_k=5, sentiment_filter="positive")
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        pipeline_config: Optional[PipelineConfig] = None,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.pipeline = NLPPipeline(pipeline_config)
        self._bm25: Optional[BM25Okapi] = None
        self._documents: List[str] = []
        self._sentiments: List[Optional[str]] = []
        self._metadata: List[Dict] = []
        self._tokenized: List[List[str]] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(
        self,
        documents: List[str],
        sentiments: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None,
    ) -> "BM25SearchEngine":
        logger.info(f"Indexing {len(documents):,} documents...")
        self._documents = documents
        self._sentiments = sentiments or [None] * len(documents)
        self._metadata = metadata or [{} for _ in documents]
        self._tokenized = [self.pipeline.process(d) for d in documents]
        self._bm25 = BM25Okapi(self._tokenized, k1=self.k1, b=self.b)
        logger.info("Index ready.")
        return self

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        sentiment_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Retrieve top-k documents by BM25 score.

        Args:
            query: Raw (unprocessed) query string.
            top_k: Maximum results to return.
            sentiment_filter: If set, only return docs with this sentiment label.

        Returns:
            List of SearchResult sorted by descending BM25 score.
        """
        if self._bm25 is None:
            raise RuntimeError("Engine not indexed. Call index() first.")

        query_tokens = self.pipeline.process(query)
        scores = self._bm25.get_scores(query_tokens)
        ranked = np.argsort(scores)[::-1]

        results: List[SearchResult] = []
        for idx in ranked:
            if len(results) >= top_k:
                break
            if scores[idx] <= 0:
                break
            sent = self._sentiments[idx]
            if sentiment_filter and sent != sentiment_filter:
                continue
            results.append(
                SearchResult(
                    doc_id=int(idx),
                    text=self._documents[idx],
                    score=float(scores[idx]),
                    sentiment=sent,
                    metadata=self._metadata[idx],
                )
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k1": self.k1,
            "b": self.b,
            "pipeline_config": self.pipeline.config,
            "documents": self._documents,
            "sentiments": self._sentiments,
            "metadata": self._metadata,
            "tokenized": self._tokenized,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"Search engine saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BM25SearchEngine":
        with open(path, "rb") as f:
            data = pickle.load(f)
        inst = cls(
            k1=data["k1"],
            b=data["b"],
            pipeline_config=data.get("pipeline_config"),
        )
        inst._documents = data["documents"]
        inst._sentiments = data["sentiments"]
        inst._metadata = data["metadata"]
        inst._tokenized = data["tokenized"]
        inst._bm25 = BM25Okapi(inst._tokenized, k1=data["k1"], b=data["b"])
        logger.info(f"Search engine loaded from {path} ({len(inst._documents):,} docs)")
        return inst
