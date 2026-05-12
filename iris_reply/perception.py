from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import tiktoken

from astrbot.api import logger

from .config import ConfigManager
from .state import StateManager

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent


@dataclass
class WindowMessage:
    sender_id: str
    sender_name: str
    content: str
    timestamp: float


class Gatekeeper:
    def __init__(self, config: ConfigManager, state: StateManager) -> None:
        self._config = config
        self._state = state

    def should_process(self, event: AstrMessageEvent) -> bool:
        message_str = event.message_str
        if not message_str:
            return False
        if message_str.startswith("/"):
            return False
        if event.is_private_chat():
            return False
        if self._state.is_muted():
            return False
        group_id = event.get_group_id()
        if not self._state.is_whitelisted(group_id):
            return False
        return True

    def quality_score(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        stripped = text.strip()
        if re.fullmatch(r"[\d\s]+", stripped):
            return 0.1
        if len(stripped) <= 2:
            return 0.2
        if re.fullmatch(r"[\W_]+", stripped):
            return 0.1
        alpha_count = sum(1 for c in stripped if c.isalnum() or "\u4e00" <= c <= "\u9fff")
        ratio = alpha_count / len(stripped) if stripped else 0
        return min(1.0, ratio)


class SlidingWindow:
    def __init__(self, config: ConfigManager) -> None:
        self._config = config
        self._windows: dict[str, deque[WindowMessage]] = {}

    def _ensure_window(self, group_id: str) -> deque[WindowMessage]:
        if group_id not in self._windows:
            self._windows[group_id] = deque(maxlen=self._config.window_size)
        return self._windows[group_id]

    def append(self, group_id: str, msg: WindowMessage) -> None:
        window = self._ensure_window(group_id)
        window.append(msg)

    def get_messages(self, group_id: str) -> list[WindowMessage]:
        window = self._ensure_window(group_id)
        return list(window)


class ContextPackager:
    def __init__(self, config: ConfigManager) -> None:
        self._config = config
        try:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Iris Reply: tiktoken init failed, falling back to char estimate: {e}")
            self._encoding = None

    def _count_tokens(self, text: str) -> int:
        if self._encoding:
            return len(self._encoding.encode(text))
        return max(1, len(text) // 2)

    def package(
        self,
        group_id: str,
        messages: list[WindowMessage],
        trigger_reason: str,
    ) -> str:
        lines = []
        for msg in messages:
            lines.append(f"[{msg.sender_name}({msg.sender_id})] {msg.content}")
        context_text = "\n".join(lines)

        total_tokens = self._count_tokens(context_text)
        max_tokens = self._config.max_token

        while total_tokens > max_tokens and lines:
            lines.pop(0)
            context_text = "\n".join(lines)
            total_tokens = self._count_tokens(context_text)

        header = f"<iris_context trigger_reason=\"{trigger_reason}\">\n"
        footer = "\n</iris_context>"
        return header + context_text + footer
