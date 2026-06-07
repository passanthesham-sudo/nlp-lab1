"""TF-IDF vectorizer (sklearn wrapper with helpers for inspection)."""
from typing import List, Tuple

import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDF:
    """
    TF-IDF vectorizer.
    Downweights ubiquitous terms and upweights discriminative ones.
    sublinear_tf=True applies log(1+tf) which helps on long reviews.
    """

    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: tuple = (1, 2),
        sublinear_tf: bool = True,
        **kwargs,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            **kwargs,
        )

    def fit(self, texts: List[str]) -> "TFIDF":
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

    def get_top_features_by_idf(self, n: int = 20) -> List[Tuple[str, float]]:
        """Return the N highest-IDF (most discriminative) features."""
        names = self.get_feature_names()
        scores = self.vectorizer.idf_
        top = scores.argsort()[-n:][::-1]
        return [(names[i], float(scores[i])) for i in top]
