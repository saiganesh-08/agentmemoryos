import re
import math
from typing import Any


# keywords that bump importance up
HIGH_IMPORTANCE_SIGNALS = [
    "error", "critical", "important", "remember", "always", "never",
    "must", "required", "deadline", "urgent", "key", "goal", "objective",
    "user", "name", "prefer", "password",  # probably shouldn't store passwords but just in case
]

LOW_IMPORTANCE_SIGNALS = [
    "maybe", "perhaps", "sometime", "random", "test", "example",
    "placeholder", "temp", "draft"
]

# tags that carry weight
TAG_WEIGHTS = {
    "core": 0.3,
    "user_pref": 0.25,
    "goal": 0.25,
    "fact": 0.15,
    "context": 0.1,
    "temp": -0.2,
    "noise": -0.3,
}


class ImportanceScorer:
    """
    Scores memory importance on 0-1 scale.
    Combines content signals, tag weights, and content length heuristics.

    Not a perfect science — these weights were tuned by running it on
    a few test agent conversations and seeing what felt right.
    """

    def __init__(self, base_score=0.5):
        self.base_score = base_score

    def score(self, content: Any, tags: list = None) -> float:
        score = self.base_score
        tags = tags or []

        text = self._extract_text(content)

        if text:
            score += self._keyword_signal(text)
            score += self._length_signal(text)

        for tag in tags:
            score += TAG_WEIGHTS.get(tag.lower(), 0.0)

        return round(max(0.01, min(1.0, score)), 3)

    def _extract_text(self, content) -> str:
        if isinstance(content, str):
            return content.lower()
        if isinstance(content, dict):
            return " ".join(str(v) for v in content.values()).lower()
        if isinstance(content, (list, tuple)):
            return " ".join(str(x) for x in content).lower()
        return str(content).lower()

    def _keyword_signal(self, text: str) -> float:
        high_hits = sum(1 for kw in HIGH_IMPORTANCE_SIGNALS if kw in text)
        low_hits = sum(1 for kw in LOW_IMPORTANCE_SIGNALS if kw in text)
        return (high_hits * 0.05) - (low_hits * 0.05)

    def _length_signal(self, text: str) -> float:
        # very short content is usually low value, long content carries more info
        # but diminishing returns after ~500 chars
        words = len(text.split())
        if words < 3:
            return -0.1
        if words < 10:
            return 0.0
        return min(0.1, math.log(words / 10) * 0.05)
