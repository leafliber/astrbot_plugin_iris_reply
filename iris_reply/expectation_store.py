from __future__ import annotations

from typing import Dict, List, Optional

from astrbot.api import logger

from .models import FollowUpExpectation


class ExpectationStore:
    def __init__(self) -> None:
        self._expectations: Dict[str, FollowUpExpectation] = {}

    def put(self, expectation: FollowUpExpectation) -> None:
        old = self._expectations.get(expectation.group_id)
        if old:
            logger.debug(
                f"Replacing expectation for group {expectation.group_id}: "
                f"{old.expectation_id} -> {expectation.expectation_id}"
            )
        self._expectations[expectation.group_id] = expectation

    def get(self, group_id: str) -> Optional[FollowUpExpectation]:
        exp = self._expectations.get(group_id)
        if exp is None:
            return None

        if exp.is_window_expired:
            logger.debug(
                f"Expectation {exp.expectation_id} for group {group_id} "
                f"window expired, removing"
            )
            del self._expectations[group_id]
            return None

        return exp

    def remove(self, group_id: str) -> Optional[FollowUpExpectation]:
        return self._expectations.pop(group_id, None)

    def has_active(self, group_id: str) -> bool:
        exp = self.get(group_id)
        return exp is not None

    def get_all(self) -> List[FollowUpExpectation]:
        expired_groups = [
            g for g, e in self._expectations.items() if e.is_window_expired
        ]
        for g in expired_groups:
            del self._expectations[g]

        return list(self._expectations.values())

    def clear(self) -> int:
        count = len(self._expectations)
        self._expectations.clear()
        return count

    @property
    def active_count(self) -> int:
        return len(self.get_all())
