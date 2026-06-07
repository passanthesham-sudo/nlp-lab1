"""Bag-of-Words vectorizer (sklearn wrapper with a consistent interface)."""
from typing import List

import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer


class BagOfWords:
    """
    Bag-of-Words vectorizer.
    Baseline that ignores term importance and document frequency.
    Compare with TFIDF to see why raw counts underperform on imbalanced vocab.
    """

    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: tuple = (1, 2),
        **kwargs,
    ) -> None:
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            **kwargs,
        )

    def fit(self, texts: List[str]) -> "BagOfWords":
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: List[str]) -> scipy.sparse.csr_matrix:
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]) -> scipy.sparse.csr_matrix:
        return self.vectorizer.fit_transform(texts)

    @property
    def vocabulary_size(self) -> int:
        return len(self.vectorizer.vocabulary_)

    def get_feature_names(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()
