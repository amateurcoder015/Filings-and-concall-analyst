from __future__ import annotations

import re
import zlib

import numpy as np


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder. No model download; used in tests and as a fallback."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                out[i, zlib.crc32(token.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class LocalEmbedder:
    """Local sentence-transformers model; downloads weights on first use."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return np.asarray(vectors, dtype=np.float32)
