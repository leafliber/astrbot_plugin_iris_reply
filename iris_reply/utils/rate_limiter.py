from __future__ import annotations

import time
import logging
from collections import defaultdict
from typing import Dict

logger = logging.getLogger("iris_reply.rate_limiter")


class RateLimiter:
    def __init__(self, max_per_hour: int = 3):
        self._max_per_hour = max_per_hour
        self._per_group_max: Dict[str, int] = {}
        self._timestamps: Dict[str, list[float]] = defaultdict(list)

    def set_group_limit(self, group_id: str, max_per_hour: int) -> None:
        self._per_group_max[group_id] = max_per_hour

    def get_limit(self, group_id: str) -> int:
        return self._per_group_max.get(group_id, self._max_per_hour)

    def is_allowed(self, group_id: str) -> bool:
        self._cleanup(group_id)
        limit = self.get_limit(group_id)
        return len(self._timestamps[group_id]) < limit

    def record(self, group_id: str) -> None:
        self._timestamps[group_id].append(time.time())

    def get_remaining(self, group_id: str) -> int:
        self._cleanup(group_id)
        limit = self.get_limit(group_id)
        return max(0, limit - len(self._timestamps[group_id]))

    def _cleanup(self, group_id: str) -> None:
        cutoff = time.time() - 3600
        self._timestamps[group_id] = [
            t for t in self._timestamps[group_id] if t > cutoff
        ]
