from __future__ import annotations

from collections import Counter
from typing import Iterable


def map_genre(tags: Iterable[str], keyword_map: dict[str, list[str]]) -> str | None:
    tag_set = {t.lower().strip() for t in tags if t}
    for genre, keywords in keyword_map.items():
        if any(keyword.lower() in tag_set for keyword in keywords):
            return genre
    return None


def score_label_confidence(tags: Iterable[str], keyword_map: dict[str, list[str]]) -> float:
    tag_counts = Counter(t.lower().strip() for t in tags if t)
    if not tag_counts:
        return 0.0

    matched = 0
    total = sum(tag_counts.values())
    for keywords in keyword_map.values():
        matched += sum(tag_counts.get(k.lower(), 0) for k in keywords)
    return matched / total if total else 0.0
