from __future__ import annotations

from .state import StateManager


class AdminCommands:
    def __init__(self, state: StateManager) -> None:
        self._state = state

    def get_status(self, group_id: str) -> str:
        return self._state.get_status_text(group_id)

    def reset_group(self, group_id: str) -> str:
        self._state.reset_group(group_id)
        return f"群 {group_id} 状态已重置"

    def set_cooldown(self, group_id: str, minutes: int) -> str:
        minutes = max(1, min(120, minutes))
        self._state.set_cooldown(group_id, minutes)
        return f"群 {group_id} 冷却已设置为 {minutes} 分钟"
