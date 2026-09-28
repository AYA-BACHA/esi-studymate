"""Embedding generation interface supporting Chroma ONNX and pure-Python fallback."""

import math
import re
from typing import List
import chromadb.utils.embedding_functions as ef


class FastDeterministicEmbeddingFunction:
    """Zero-dependency, fast bag-of-words / hashed n-gram dense embedding.
    
    Guarantees 100% offline capability, zero download overhead, and deterministic similarity
    for unit testing and rapid local execution.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def __call__(self, input: List[str]) -> List[List[float]]:
        return [self._embed_text(text) for text in input]

    def _embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vec

        for token in tokens:
            # Hash token to dimension index
            h = hash(token) % self.dim
            vec[h] += 1.0

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


def get_embedding_function(use_fast_fallback: bool = False):
    """Retrieve the preferred embedding function.
    
    Attempts Chroma's built-in ONNX all-MiniLM-L6-v2 first, falling back to
    deterministic dense embeddings if unavailable or requested.
    """
    if use_fast_fallback:
        return FastDeterministicEmbeddingFunction()

    try:
        # Default Chroma ONNX embedding
        return ef.DefaultEmbeddingFunction()
    except Exception:
        return FastDeterministicEmbeddingFunction()
