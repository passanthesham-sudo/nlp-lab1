FROM python:3.11-slim

WORKDIR /app

# System build deps (needed by some NLTK/gensim native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-caching friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data so the container starts without internet access
RUN python -c "\
import nltk; \
nltk.download('punkt_tab', quiet=True); \
nltk.download('stopwords', quiet=True); \
nltk.download('wordnet', quiet=True); \
nltk.download('omw-1.4', quiet=True); \
"

# Copy source
COPY src/ ./src/
COPY params.yaml .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
