from __future__ import annotations

from typing import TYPE_CHECKING

from astrbot.api import logger

from .config import ConfigManager
from .state import GroupState, StateManager

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent


class TriggerEngine:
    def __init__(self, config: ConfigManager, state: StateManager) -> None:
        self._config = config
        self._state = state

    def check_follow_up(self, group_id: str, sender_id: str, message: str) -> bool:
        data = self._state.get_state(group_id)
        if data.state == GroupState.COOLDOWN:
            return False
        if self._state.is_muted():
            return False
        return self._state.match_follow_up(group_id, sender_id, message)

    def check_sampling(self, group_id: str) -> bool:
        return self._state.should_trigger_sampling(group_id)

    def evaluate(self, event: AstrMessageEvent) -> str | None:
        if not self._config.enabled:
            return None

        group_id = event.get_group_id()
        if not group_id:
            return None

        sender_id = event.get_sender_id()
        message_str = event.message_str or ""

        if self.check_follow_up(group_id, sender_id, message_str):
            logger.debug(f"Iris Reply: follow-up trigger for group {group_id}")
            return "follow_up"

        self._state.increment_msg_count(group_id)

        if self.check_sampling(group_id):
            logger.debug(f"Iris Reply: sampling trigger for group {group_id}")
            self._state.reset_sampling(group_id)
            return "sampling"

        return None
