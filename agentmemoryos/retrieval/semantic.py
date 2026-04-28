import numpy as np
import re
from typing import Callable, Optional


# pluggable embedding function signature: (text: str) -> np.ndarray
# by default we use a simple bag-of-words TF-IDF style embedding
# swap in sentence-transformers or openai embeddings for better results

def _tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"\b[a-z][a-z0-9]*\b", text)
    return tokens


def _build_vocab(corpus: list[str]) -> dict[str, int]:
    vocab = {}
    for text in corpus:
        for tok in _tokenize(text):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def _bow_embed(text: str, vocab: dict) -> np.ndarray:
    vec = np.zeros(len(vocab))
    for tok in _tokenize(text):
        if tok in vocab:
            vec[vocab[tok]] += 1
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SemanticIndex:
    """
    Lightweight semantic search over memory items.

    Uses bag-of-words embeddings by default — no external dependencies.
    For production use, pass in a real embed_fn (sentence-transformers, openai, etc.)

    Usage:
        index = SemanticIndex()
        index.add("mem_1", "the user prefers dark mode")
        index.add("mem_2", "meeting scheduled for friday")
        results = index.search("what are the user preferences", top_k=3)
    """

    def __init__(self, embed_fn: Optional[Callable] = None):
        self.embed_fn = embed_fn  # if None, falls back to BoW
        self._entries: dict[str, str] = {}  # key -> text
        self._vectors: dict[str, np.ndarray] = {}
        self._vocab: dict[str, int] = {}
        self._dirty = False  # vocab needs rebuilding

    def add(self, key: str, text: str):
        self._entries[key] = text
        self._dirty = True

    def remove(self, key: str):
        self._entries.pop(key, None)
        self._vectors.pop(key, None)
        self._dirty = True

    def _rebuild(self):
        if self.embed_fn:
            for key, text in self._entries.items():
                if key not in self._vectors:
                    self._vectors[key] = self.embed_fn(text)
        else:
            self._vocab = _build_vocab(list(self._entries.values()))
            self._vectors = {
                key: _bow_embed(text, self._vocab)
                for key, text in self._entries.items()
            }
        self._dirty = False

    def search(self, query: str, top_k=5, min_score=0.0) -> list[tuple[str, float]]:
        if not self._entries:
            return []

        if self._dirty:
            self._rebuild()

        if self.embed_fn:
            q_vec = self.embed_fn(query)
        else:
            q_vec = _bow_embed(query, self._vocab)

        scores = []
        for key, vec in self._vectors.items():
            sim = cosine_similarity(q_vec, vec)
            if sim >= min_score:
                scores.append((key, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def __len__(self):
        return len(self._entries)
