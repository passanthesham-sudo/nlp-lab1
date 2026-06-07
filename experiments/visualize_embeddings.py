"""
Train Word2Vec embeddings and visualize them with PCA and t-SNE.

What this demonstrates
----------------------
- BoW and TF-IDF represent each word as an independent orthogonal dimension.
  'good' and 'great' have zero similarity.
- Word2Vec learns dense vectors where semantically similar words cluster
  together — food domain words cluster separately from quality words, etc.

Usage
-----
    python experiments/visualize_embeddings.py
    python experiments/visualize_embeddings.py --sample 30000
"""
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.pipeline import NLPPipeline
from src.vectorization.embeddings import WordEmbeddings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

FOOD_WORDS = [
    "food", "taste", "flavor", "fresh", "delicious", "organic", "natural",
    "sweet", "salty", "spicy", "bitter", "rich", "crispy", "juicy",
]
SENTIMENT_WORDS = [
    "good", "great", "excellent", "amazing", "perfect", "love",
    "bad", "terrible", "awful", "disgusting", "horrible", "disappointed",
]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--amazon-path", default="data/raw/Reviews.csv")
    parser.add_argument("--sample", type=int, default=20_000)
    parser.add_argument("--output-dir", default="outputs/embeddings")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    amazon_path = Path(args.amazon_path)
    if not amazon_path.exists():
        logger.error(f"Dataset not found: {amazon_path}")
        return

    df = pd.read_csv(str(amazon_path)).dropna(subset=["Text"])
    df = df.sample(min(args.sample, len(df)), random_state=42)

    pipeline = NLPPipeline(NLPPipeline.REVIEWS_CONFIG)
    tokenized = pipeline.batch_process(df["Text"].tolist(), return_strings=False)

    emb = WordEmbeddings(vector_size=100, window=5, min_count=5, epochs=10)
    emb.train(tokenized)

    # Semantic similarity examples
    print("\n" + "=" * 55)
    print("SEMANTIC SIMILARITY (what BoW/TF-IDF cannot capture)")
    print("=" * 55)
    for word in ["good", "food", "flavor", "love", "terrible"]:
        similar = emb.most_similar(word, topn=5)
        if similar:
            top = ", ".join(f"{w}({s:.2f})" for w, s in similar)
            print(f"  '{word}' -> {top}")
    print("=" * 55)

    # Visualize all top words
    emb.visualize(
        method="tsne",
        title="Amazon Reviews — Word Embeddings (t-SNE)",
        save_path=str(out / "amazon_tsne.png"),
    )
    emb.visualize(
        method="pca",
        title="Amazon Reviews — Word Embeddings (PCA)",
        save_path=str(out / "amazon_pca.png"),
    )

    # Visualize food + sentiment cluster together
    highlight = [
        w for w in FOOD_WORDS + SENTIMENT_WORDS if w in emb.model.wv
    ]
    if highlight:
        emb.visualize(
            words=highlight,
            method="pca",
            title="Food vs Sentiment Words (PCA) — clusters absent in BoW",
            save_path=str(out / "food_vs_sentiment_pca.png"),
        )

    emb.save(str(out / "word2vec_amazon.model"))
    logger.info(f"Saved plots and model to {out}/")


if __name__ == "__main__":
    main()
