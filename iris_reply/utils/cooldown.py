from __future__ import annotations

import time
import logging
from collections import defaultdict
from typing import Dict

logger = logging.getLogger("iris_reply.cooldown")


class CooldownManager:
    def __init__(self, default_seconds: int = 300):
        self._default_seconds = default_seconds
        self._per_group: Dict[str, int] = {}
        self._last_reply: Dict[str, float] = defaultdict(float)

    def set_group_cooldown(self, group_id: str, seconds: int) -> None:
        self._per_group[group_id] = seconds

    def get_cooldown(self, group_id: str) -> int:
        return self._per_group.get(group_id, self._default_seconds)

    def is_on_cooldown(self, group_id: str) -> bool:
        cd = self.get_cooldown(group_id)
        elapsed = time.time() - self._last_reply.get(group_id, 0)
        return elapsed < cd

    def mark_reply(self, group_id: str) -> None:
        self._last_reply[group_id] = time.time()

    def get_remaining(self, group_id: str) -> float:
        cd = self.get_cooldown(group_id)
        elapsed = time.time() - self._last_reply.get(group_id, 0)
        remaining = cd - elapsed
        return max(0.0, remaining)
