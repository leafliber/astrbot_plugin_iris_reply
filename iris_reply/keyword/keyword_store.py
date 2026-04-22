from __future__ import annotations

import logging
from typing import Any

from iris_reply.storage.store import Store
from iris_reply.models.models import KeywordSource

logger = logging.getLogger("iris_reply.keyword_store")


class KeywordStore:
    def __init__(self, store: Store, global_static_keywords: list[str] | None = None):
        self._store = store
        self._global_static = global_static_keywords or []
        self._dynamic_cache: dict[str, list[str]] = {}

    def get_all_keywords(self, group_id: str) -> list[str]:
        static = self.get_static_keywords(group_id)
        dynamic = self.get_dynamic_keywords(group_id)
        seen = set()
        result = []
        for kw in static + dynamic:
            if kw not in seen:
                seen.add(kw)
                result.append(kw)
        return result

    def get_static_keywords(self, group_id: str) -> list[str]:
        db_keywords = self._store.get_static_keywords(group_id)
        seen = set(db_keywords)
        for kw in self._global_static:
            if kw not in seen:
                seen.add(kw)
                db_keywords.append(kw)
        return db_keywords

    def add_static_keyword(self, group_id: str, keyword: str) -> bool:
        return self._store.add_static_keyword(group_id, keyword)

    def remove_static_keyword(self, group_id: str, keyword: str) -> bool:
        return self._store.remove_static_keyword(group_id, keyword)

    def get_dynamic_keywords(self, group_id: str) -> list[str]:
        cached = self._dynamic_cache.get(group_id)
        if cached is not None:
            return cached
        keywords = self._store.get_dynamic_keywords(group_id)
        self._dynamic_cache[group_id] = keywords
        return keywords

    def update_dynamic_keywords(self, group_id: str, keywords: list[tuple[str, KeywordSource, float]]) -> None:
        self._store.set_dynamic_keywords(group_id, keywords)
        self._dynamic_cache[group_id] = [kw for kw, _, _ in keywords]
        logger.info("已更新群 %s 的动态关键词，共 %d 个", group_id, len(keywords))

    def invalidate_cache(self, group_id: str | None = None) -> None:
        if group_id:
            self._dynamic_cache.pop(group_id, None)
        else:
            self._dynamic_cache.clear()
