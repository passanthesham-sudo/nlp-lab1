"""
Word2Vec embeddings with PCA/t-SNE visualization.

Classical vectors (BoW, TF-IDF) treat every word as an independent dimension.
Word2Vec learns dense vectors where similar words cluster together — this script
makes that visible.
"""
import logging
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


class WordEmbeddings:
    """Gensim Word2Vec with visualization helpers."""

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 5,
        workers: int = 4,
        epochs: int = 5,
        sg: int = 0,  # 0=CBOW, 1=Skip-gram
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.sg = sg
        self.model: Optional[Word2Vec] = None

    def train(self, tokenized_docs: List[List[str]]) -> "WordEmbeddings":
        logger.info(f"Training Word2Vec on {len(tokenized_docs):,} documents...")
        self.model = Word2Vec(
            sentences=tokenized_docs,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            sg=self.sg,
        )
        logger.info(f"Vocabulary size: {len(self.model.wv.key_to_index):,}")
        return self

    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        if self.model and word in self.model.wv:
            return self.model.wv.most_similar(word, topn=topn)
        return []

    def document_vector(self, tokens: List[str]) -> np.ndarray:
        """Mean-pool word vectors for a document (for downstream classifiers)."""
        vecs = [self.model.wv[t] for t in tokens if self.model and t in self.model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(self.vector_size)

    def visualize(
        self,
        words: Optional[List[str]] = None,
        method: str = "tsne",
        save_path: Optional[str] = None,
        title: str = "Word Embeddings",
    ) -> None:
        if self.model is None:
            raise RuntimeError("Train the model first.")

        if words is None:
            words = list(self.model.wv.key_to_index.keys())[:150]

        words = [w for w in words if w in self.model.wv]
        if not words:
            logger.warning("No valid words to visualize.")
            return

        vectors = np.array([self.model.wv[w] for w in words])

        if method == "pca":
            reducer = PCA(n_components=2)
            reduced = reducer.fit_transform(vectors)
        else:
            perplexity = min(30, max(5, len(words) - 1))
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            reduced = reducer.fit_transform(vectors)

        fig, ax = plt.subplots(figsize=(14, 10))
        ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.4, s=15)
        for i, word in enumerate(words):
            ax.annotate(word, (reduced[i, 0], reduced[i, 1]), fontsize=7, alpha=0.75)
        ax.set_title(title)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close(fig)

    def save(self, path: str) -> None:
        if self.model:
            self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "WordEmbeddings":
        inst = cls()
        inst.model = Word2Vec.load(path)
        return inst
