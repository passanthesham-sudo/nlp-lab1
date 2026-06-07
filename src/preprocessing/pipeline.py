"""
Configurable NLP preprocessing pipeline supporting both long-form reviews
and short-form social media text.

The two preset configs (REVIEWS_CONFIG, TWITTER_CONFIG) deliberately produce
different behavior on the same input — this is the core engineering lesson of the lab.
"""
import re
import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional

import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

logger = logging.getLogger(__name__)


def _ensure_nltk_data() -> None:
    for name in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        nltk.download(name, quiet=True)


_ensure_nltk_data()


@dataclass
class PipelineConfig:
    lowercase: bool = True
    remove_html: bool = True
    remove_urls: bool = True
    remove_mentions: bool = False
    remove_hashtags: bool = False
    keep_hashtag_text: bool = True
    remove_punctuation: bool = True
    remove_numbers: bool = False
    remove_stopwords: bool = True
    stemming: bool = False
    lemmatization: bool = True
    min_token_length: int = 2
    language: str = "english"
    use_tweet_tokenizer: bool = False


class NLPPipeline:
    """
    Configurable NLP preprocessing pipeline.

    Handles the full sequence: clean → tokenize → filter → normalize.
    Designed to be instantiated with domain-specific configs rather than
    rewritten per dataset.
    """

    # Long-form product reviews: HTML may be present, lemmatization preferred
    REVIEWS_CONFIG = PipelineConfig(
        remove_html=True,
        remove_mentions=False,
        stemming=False,
        lemmatization=True,
        use_tweet_tokenizer=False,
    )

    # Short noisy tweets: no HTML, mentions stripped, stemming preferred (short context)
    TWITTER_CONFIG = PipelineConfig(
        remove_html=False,
        remove_mentions=True,
        remove_hashtags=False,
        keep_hashtag_text=True,
        stemming=True,
        lemmatization=False,
        use_tweet_tokenizer=True,
    )

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._stemmer = PorterStemmer() if self.config.stemming else None
        self._lemmatizer = WordNetLemmatizer() if self.config.lemmatization else None
        self._stopwords = (
            set(stopwords.words(self.config.language))
            if self.config.remove_stopwords
            else set()
        )
        self._tweet_tokenizer = (
            TweetTokenizer(strip_handles=True, reduce_len=True)
            if self.config.use_tweet_tokenizer
            else None
        )

    # ------------------------------------------------------------------
    # Individual steps
    # ------------------------------------------------------------------

    def clean(self, text: str) -> str:
        """Apply string-level cleaning (HTML, URLs, punctuation, etc.)."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        if self.config.remove_html:
            text = BeautifulSoup(text, "lxml").get_text(separator=" ")

        if self.config.lowercase:
            text = text.lower()

        if self.config.remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        if self.config.remove_mentions:
            text = re.sub(r"@\w+", " ", text)

        if self.config.remove_hashtags:
            text = re.sub(r"#\w+", " ", text)
        elif self.config.keep_hashtag_text:
            text = re.sub(r"#(\w+)", r"\1", text)

        if self.config.remove_numbers:
            text = re.sub(r"\d+", " ", text)

        if self.config.remove_punctuation:
            text = re.sub(r"[^\w\s]", " ", text)

        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        if self._tweet_tokenizer:
            return self._tweet_tokenizer.tokenize(text)
        return word_tokenize(text)

    def filter_tokens(self, tokens: List[str]) -> List[str]:
        return [
            t for t in tokens
            if t not in self._stopwords and len(t) >= self.config.min_token_length
        ]

    def normalize_tokens(self, tokens: List[str]) -> List[str]:
        if self._stemmer:
            return [self._stemmer.stem(t) for t in tokens]
        if self._lemmatizer:
            return [self._lemmatizer.lemmatize(t) for t in tokens]
        return tokens

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def process(self, text: str) -> List[str]:
        """Full pipeline → list of tokens."""
        return self.normalize_tokens(
            self.filter_tokens(
                self.tokenize(
                    self.clean(text)
                )
            )
        )

    def process_to_string(self, text: str) -> str:
        """Full pipeline → space-joined string (sklearn-compatible)."""
        return " ".join(self.process(text))

    def batch_process(
        self, texts: List[str], return_strings: bool = True
    ) -> List:
        fn = self.process_to_string if return_strings else self.process
        return [fn(t) for t in texts]
