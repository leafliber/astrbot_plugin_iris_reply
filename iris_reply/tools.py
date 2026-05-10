from __future__ import annotations


class ToolContext:
    def __init__(self) -> None:
        self._current_group_id: str | None = None

    def set_context(self, group_id: str) -> None:
        self._current_group_id = group_id

    def clear_context(self) -> None:
        self._current_group_id = None

    @property
    def current_group_id(self) -> str | None:
        return self._current_group_id
