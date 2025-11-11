# llm_radar/retrieval/embedder.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class Embedder(ABC):
    @abstractmethod
    def dim(self) -> int: ...
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:  # (N, D)
        ...

class LocalSBERTEmbedder(Embedder):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        _vec = self.model.encode(["warmup"], convert_to_numpy=True, normalize_embeddings=True)
        self._dim = _vec.shape[1]

    def dim(self) -> int: return int(self._dim)
    def name(self) -> str: return f"sbert:{self.model_name}"

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
            batch_size=64, show_progress_bar=False
        )
        return vecs.astype("float32")

class TFIDFEmbedder(Embedder):
    def __init__(self, max_features: int = 4096):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self._dim = max_features

    def dim(self) -> int: return int(self._dim)
    def name(self) -> str: return f"tfidf:{self._dim}"

    def encode(self, texts: List[str]) -> np.ndarray:
        from scipy.sparse import spmatrix
        if not hasattr(self.vectorizer, "vocabulary_"):
            X = self.vectorizer.fit_transform(texts)
        else:
            X = self.vectorizer.transform(texts)
        if isinstance(X, spmatrix):
            X = X.toarray()
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        self._dim = X.shape[1]
        return X.astype("float32")