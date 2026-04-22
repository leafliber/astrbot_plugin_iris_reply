from __future__ import annotations

import logging
from difflib import SequenceMatcher

from iris_reply.models.models import KeywordMatch, KeywordMatchType, KeywordSource

logger = logging.getLogger("iris_reply.keyword_matcher")


class KeywordMatcher:
    def __init__(self, fuzzy_threshold: float = 0.8):
        self._fuzzy_threshold = fuzzy_threshold

    def match(self, message: str, keywords: list[str], source: KeywordSource = KeywordSource.STATIC) -> list[KeywordMatch]:
        if not message or not keywords:
            return []
        results: list[KeywordMatch] = []
        message_lower = message.lower()

        for keyword in keywords:
            keyword_lower = keyword.lower()

            exact_pos = message_lower.find(keyword_lower)
            if exact_pos != -1:
                results.append(KeywordMatch(
                    keyword=keyword,
                    match_type=KeywordMatchType.EXACT,
                    confidence=1.0,
                    position=exact_pos,
                    source=source,
                ))
                continue

            fuzzy_result = self._fuzzy_match(message_lower, keyword_lower)
            if fuzzy_result is not None:
                pos, confidence = fuzzy_result
                if confidence >= self._fuzzy_threshold:
                    results.append(KeywordMatch(
                        keyword=keyword,
                        match_type=KeywordMatchType.FUZZY,
                        confidence=confidence,
                        position=pos,
                        source=source,
                    ))

        results.sort(key=lambda m: m.confidence, reverse=True)
        return results

    def _fuzzy_match(self, message: str, keyword: str) -> tuple[int, float] | None:
        kw_len = len(keyword)
        if kw_len == 0 or len(message) < kw_len:
            return None

        best_ratio = 0.0
        best_pos = -1
        step = max(1, kw_len // 3)

        for i in range(0, len(message) - kw_len + 1, step):
            segment = message[i : i + kw_len]
            ratio = SequenceMatcher(None, keyword, segment).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = i

        if best_pos >= 0:
            return best_pos, best_ratio
        return None
