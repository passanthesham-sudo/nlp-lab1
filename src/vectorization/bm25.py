"""
BM25 vectorizer — used for retrieval (not classification).

BM25 normalizes by document length (parameter b) and saturates term frequency
(parameter k1), making it consistently better than TF-IDF for search over
variable-length documents like product reviews.
"""
from typing import Any, List

import numpy as np
from rank_bm25 import BM25Okapi


class BM25Vectorizer:
    """
    BM25 wrapper with a sklearn-style fit/transform interface.
    Input must be pre-tokenized (List[List[str]]).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._bm25: BM25Okapi | None = None

    def fit(self, tokenized_docs: List[List[str]]) -> "BM25Vectorizer":
        self._bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        return self

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        if self._bm25 is None:
            raise RuntimeError("Call fit() before get_scores().")
        return self._bm25.get_scores(query_tokens)

    def get_top_n(
        self, query_tokens: List[str], documents: List[Any], n: int = 10
    ) -> List[Any]:
        if self._bm25 is None:
            raise RuntimeError("Call fit() before get_top_n().")
        return self._bm25.get_top_n(query_tokens, documents, n=n)
