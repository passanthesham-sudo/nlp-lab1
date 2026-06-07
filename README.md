# NLP Intelligence System 

Sentiment classifier + BM25 search engine over **Amazon Fine Food Reviews** and **Sentiment140**, tracked with MLflow, versioned with DVC, and deployed as a Dockerized FastAPI service.

---

## Project structure

```
.
├── src/
│   ├── preprocessing/pipeline.py     # Configurable NLP pipeline (clean→tokenize→normalize)
│   ├── vectorization/
│   │   ├── bow.py                    # Bag-of-Words (CountVectorizer)
│   │   ├── tfidf.py                  # TF-IDF with IDF inspection helpers
│   │   ├── bm25.py                   # BM25 (rank_bm25 wrapper)
│   │   └── embeddings.py             # Word2Vec + PCA/t-SNE visualization
│   ├── models/sentiment.py           # Classifier (LR / SVC) with MLflow tracking
│   ├── search/engine.py              # BM25 search engine with sentiment filter
│   └── api/main.py                   # FastAPI service
├── experiments/
│   ├── train.py                      # Main training script
│   ├── compare_vectorizers.py        # BoW vs TF-IDF vs BM25 comparison
│   └── visualize_embeddings.py       # Word2Vec visualization
├── data/raw/                         # Place datasets here (gitignored, DVC-tracked)
├── models/                           # Saved model artefacts (gitignored)
├── metrics/                          # JSON metrics output
├── outputs/embeddings/               # Plots and saved embedding models
├── params.yaml                       # All tunable hyperparameters
├── dvc.yaml                          # Reproducible pipeline
├── Dockerfile
└── docker-compose.yml
```

---

## 1. Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.10+ | [python.org](https://python.org) |
| Docker + Docker Compose | 24+ | [docker.com](https://docker.com) |
| Git | any | |
| DVC | 3.x | `pip install dvc` |

---

## 2. Setup

### 2.1 Clone and install

```bash
git clone <your-repo-url>
cd <repo>

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2.2 Download datasets

**Option A — Kaggle CLI (recommended)**

```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/

# Amazon Fine Food Reviews (~250 MB)
kaggle datasets download -d snap/amazon-fine-food-reviews -p data/raw --unzip

# Sentiment140 (~80 MB)
kaggle datasets download -d kazanova/sentiment140 -p data/raw --unzip
```

**Option B — Manual download**

| Dataset | URL | Save as |
|---------|-----|---------|
| Amazon Fine Food Reviews | https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews | `data/raw/Reviews.csv` |
| Sentiment140 | https://www.kaggle.com/datasets/kazanova/sentiment140 | `data/raw/training.1600000.processed.noemoticon.csv` |

### 2.3 Initialize DVC

```bash
git init          # if not already a git repo
dvc init
git add .dvc .gitignore
git commit -m "init DVC"

# (Optional) add a local DVC remote to cache large files
dvc remote add -d localcache /tmp/dvc-cache
```

---

## 3. Training

### 3.1 Run everything in one command (DVC pipeline)

```bash
dvc repro
```

This runs all four stages in dependency order:
1. `train_amazon` — trains BoW×LR, BoW×SVC, TF-IDF×LR, TF-IDF×SVC on Amazon data; saves best model + search index
2. `train_sentiment140` — same on tweets
3. `compare_vectorizers` — side-by-side metrics table
4. `visualize_embeddings` — saves PNGs to `outputs/embeddings/`

### 3.2 Train manually (without DVC)

```bash
# Full training (both datasets)
python experiments/train.py

# Quick test run with smaller samples
python experiments/train.py --amazon-sample 5000 --s140-sample 10000

# Amazon only
python experiments/train.py --skip-sentiment140

# Custom dataset paths
python experiments/train.py \
  --amazon-path /path/to/Reviews.csv \
  --sentiment140-path /path/to/sentiment140.csv
```

### 3.3 Run individual experiments

```bash
# Compare BoW vs TF-IDF vs BM25
python experiments/compare_vectorizers.py --sample 10000

# Word embedding visualization
python experiments/visualize_embeddings.py --sample 20000
```

### 3.4 View MLflow results

```bash
mlflow ui
# Open http://localhost:5000
```

---

## 4. Running the API

### Option A — Docker Compose (recommended)

> **Requires trained models in `models/` first** (run Step 3 above).

```bash
docker-compose up --build
```

Services:
| Service | URL |
|---------|-----|
| FastAPI | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

### Option B — Local (no Docker)

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
# Open http://localhost:8000/docs
```

---

## 5. API endpoints

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "classifier_loaded": true,
  "search_engine_loaded": true,
  "search_index_size": 49823
}
```

---

### `POST /classify` — Sentiment classification

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "This product is absolutely amazing, I love it!",
      "Terrible quality, waste of money."
    ],
    "domain": "reviews"
  }'
```

```json
{
  "predictions": ["positive", "negative"],
  "probabilities": [
    {"negative": 0.0312, "positive": 0.9688},
    {"negative": 0.9741, "positive": 0.0259}
  ]
}
```

Use `"domain": "twitter"` for tweet-style input (applies Twitter preprocessing config).

---

### `POST /search` — BM25 document search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "gluten free organic snacks",
    "top_k": 5,
    "sentiment_filter": "positive"
  }'
```

```json
{
  "results": [
    {
      "doc_id": 12345,
      "text": "These gluten-free crackers are incredible...",
      "score": 8.4321,
      "sentiment": "positive",
      "metadata": {}
    }
  ],
  "total_found": 5,
  "query": "gluten free organic snacks"
}
```

Set `"sentiment_filter": null` (or omit) to search all documents regardless of sentiment.

---

### `GET /experiments` — List MLflow experiments

```bash
curl http://localhost:8000/experiments
```

---

## 6. Reproducibility with DVC

```bash
# Reproduce the full pipeline from scratch
dvc repro

# Check what has changed (which stages are stale)
dvc status

# Push data to remote (after configuring a remote)
dvc push

# Pull data on a new machine
dvc pull
```

Change any value in `params.yaml` (e.g., `training.amazon.sample_size`) and run `dvc repro` — only affected stages rerun.

---

## 7. Key engineering decisions

### Why different preprocessing configs per dataset?

| Setting | Amazon Reviews | Sentiment140 |
|---------|---------------|--------------|
| `remove_html` | `True` — reviews contain HTML entities | `False` — plain text |
| `lemmatization` | `True` — long text benefits from root forms | `False` |
| `stemming` | `False` | `True` — tweets are short; aggressive reduction helps |
| `use_tweet_tokenizer` | `False` | `True` — preserves emoticons, handles @handles |

### Why does TF-IDF outperform BoW?

Amazon reviews are long and domain-specific. Words like "product", "item", and "review" appear in nearly every document and dominate BoW vectors. TF-IDF down-weights these and amplifies discriminative terms like "rancid", "extraordinary", or "defective".

### Why is BM25 better for search than TF-IDF?

BM25 adds two corrections TF-IDF lacks:
- **Term frequency saturation** (k1): prevents a word appearing 100× from dominating over 10×.
- **Document length normalization** (b): a short review matching your query scores higher than a 5,000-word review with the same word count.

### What do word embeddings capture that BoW/TF-IDF cannot?

BoW gives `good` and `great` zero similarity. Word2Vec learns that they cluster together because they appear in similar contexts. The t-SNE plot in `outputs/embeddings/` shows food-domain words clustering away from sentiment words — structure that's invisible in a 50,000-dim sparse vector.

---

## 8. Tuning hyperparameters

All hyperparameters live in `params.yaml`. DVC detects changes and reruns only affected stages.

```yaml
training:
  amazon:
    sample_size: 50000    # increase for better accuracy
    max_features: 50000   # vocabulary size

search:
  bm25:
    k1: 1.5               # term saturation  (1.2–2.0 typical)
    b: 0.75               # length norm      (0.0=off, 1.0=full)
```

---

## 9. Running tests

```bash
pytest tests/ -v
```

> Tests are not included in this submission scope — add unit tests for the pipeline and API endpoints as a bonus.

---

## 10. Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: src` | Run from project root, or `export PYTHONPATH=.` |
| `LookupError: Resource punkt_tab not found` | `python -c "import nltk; nltk.download('punkt_tab')"` |
| `503 Classifier not loaded` | Run training first: `python experiments/train.py` |
| Docker build fails on `gcc` | Ensure Docker daemon is running and has internet access |
| MLflow runs not showing | Set `MLFLOW_TRACKING_URI=./mlruns` or run `mlflow ui` from project root |
