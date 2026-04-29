from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Final, Optional

from astrbot.api import logger

from .bounded_dict import BoundedDict
from .models import CooldownState

DEFAULT_DURATION_MINUTES: Final[int] = 20
MIN_DURATION_MINUTES: Final[int] = 5
MAX_DURATION_MINUTES: Final[int] = 180
MAX_TRACKED_GROUPS: Final[int] = 1000

_DURATION_PATTERN: Final = re.compile(
    r"^(\d+)\s*(m|min|分钟?|h|hour|小时)?$",
    re.IGNORECASE,
)


def parse_duration(raw: str) -> Optional[int]:
    raw = raw.strip()
    if not raw:
        return None

    match = _DURATION_PATTERN.match(raw)
    if not match:
        return None

    value = int(match.group(1))
    unit = (match.group(2) or "m").lower()

    if unit in ("h", "hour", "小时"):
        return value * 60
    return value


class CooldownManager:
    def __init__(
        self,
        default_duration: int = DEFAULT_DURATION_MINUTES,
        max_groups: int = MAX_TRACKED_GROUPS,
    ) -> None:
        self._default_duration = default_duration
        self._states: BoundedDict[str, CooldownState] = BoundedDict(max_size=max_groups)

    @property
    def default_duration(self) -> int:
        return self._default_duration

    def activate(
        self,
        group_id: str,
        duration_minutes: Optional[int] = None,
        reason: Optional[str] = None,
        initiated_by: str = "user",
    ) -> str:
        duration = duration_minutes or self._default_duration

        if duration < MIN_DURATION_MINUTES:
            return f"冷却时间不能少于 {MIN_DURATION_MINUTES} 分钟"
        if duration > MAX_DURATION_MINUTES:
            return f"冷却时间不能超过 {MAX_DURATION_MINUTES} 分钟（{MAX_DURATION_MINUTES // 60} 小时）"

        now = datetime.now(timezone.utc)
        state = CooldownState(
            group_id=group_id,
            started_at=now,
            expires_at=now + timedelta(minutes=duration),
            initiated_by=initiated_by,
            reason=reason,
        )
        self._states[group_id] = state

        logger.info(
            f"Cooldown activated: group={group_id}, duration={duration}min, "
            f"initiated_by={initiated_by}, reason={reason}"
        )

        reason_line = f"\n原因：{reason}" if reason else ""
        return (
            f"⏸️ 已进入冷却模式（{duration}分钟）{reason_line}\n"
            f"期间我将暂停主动回复，仅响应@消息和指令"
        )

    def deactivate(self, group_id: str) -> str:
        if group_id in self._states:
            del self._states[group_id]
            logger.info(f"Cooldown deactivated: group={group_id}")
            return "▶️ 已退出冷却模式，恢复正常服务"
        return "当前群聊未处于冷却模式"

    def get_status(self, group_id: str) -> Optional[CooldownState]:
        state = self._states.get(group_id)
        if state is None:
            return None

        if not state.is_active:
            del self._states[group_id]
            logger.debug(f"Cooldown expired and cleaned: group={group_id}")
            return None

        return state

    def get_state(self, group_id: str) -> Optional[CooldownState]:
        return self.get_status(group_id)

    def is_active(self, group_id: str) -> bool:
        return self.get_status(group_id) is not None

    @property
    def active_count(self) -> int:
        active = [k for k, v in self._states.items() if v.is_active]
        return len(active)

    def format_status(self, group_id: str) -> str:
        state = self.get_status(group_id)
        if state is None:
            return "当前群聊未处于冷却模式"

        initiator = "用户" if state.initiated_by == "user" else "系统"
        reason_line = f"\n原因：{state.reason}" if state.reason else ""
        return (
            f"⏸️ 冷却模式中\n"
            f"触发者：{initiator}\n"
            f"剩余时间：{state.format_remaining()}\n"
            f"到期时间：{state.format_expires_at_local()}"
            f"{reason_line}"
        )
