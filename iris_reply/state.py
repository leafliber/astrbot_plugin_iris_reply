from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from astrbot.api import logger

from .config import ConfigManager
from .prompts import (
    BACKOFF_BASE,
    DEFAULT_LEVEL,
    MAX_BACKOFF_LEVEL,
    VALID_LEVELS,
    WILLINGNESS_THRESHOLD_ADJUST,
)


class GroupState(Enum):
    IDLE = "idle"
    COOLDOWN = "cooldown"
    FOLLOWING = "following"


@dataclass
class FollowUpEntry:
    user_ids: set[str] = field(default_factory=set)
    user_ttls: dict[str, float] = field(default_factory=dict)


@dataclass
class GroupStateData:
    state: GroupState = GroupState.IDLE
    cooldown_until: float = 0.0
    following_since: float = 0.0
    msg_count: int = 0
    last_sample_time: float = 0.0
    backoff_level: int = 0
    auto_adjust_factor: float = 1.0
    consecutive_replies: int = 0
    skip_history: deque = field(default_factory=lambda: deque(maxlen=20))
    follow_up: FollowUpEntry = field(default_factory=FollowUpEntry)
    willingness: str = DEFAULT_LEVEL
    boost_initial: float = 1.0
    boost_set_at: float = 0.0
    boost_until: float = 0.0
    dirty: bool = False


class StateManager:
    N_MAX = 120
    T_MAX = 300

    def __init__(self, config: ConfigManager) -> None:
        self._config = config
        self._groups: dict[str, GroupStateData] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._dirty_groups: set[str] = set()
        self._global_lock = asyncio.Lock()
        self._whitelist: set[str] = set()

    def get_lock(self, group_id: str) -> asyncio.Lock:
        if group_id not in self._locks:
            self._locks[group_id] = asyncio.Lock()
        return self._locks[group_id]

    def _ensure_group(self, group_id: str) -> GroupStateData:
        if group_id not in self._groups:
            self._groups[group_id] = GroupStateData(
                last_sample_time=time.time(),
            )
        return self._groups[group_id]

    def get_state(self, group_id: str) -> GroupStateData:
        data = self._ensure_group(group_id)
        now = time.time()
        if data.state == GroupState.COOLDOWN and now >= data.cooldown_until:
            data.state = GroupState.IDLE
            data.dirty = True
            self._dirty_groups.add(group_id)
        if data.state == GroupState.FOLLOWING:
            self._cleanup_expired_follow(group_id, data, now)
            if not data.follow_up.user_ids:
                data.state = GroupState.IDLE
                data.dirty = True
                self._dirty_groups.add(group_id)
        return data

    def is_muted(self) -> bool:
        now = time.localtime()
        current_mins = now.tm_hour * 60 + now.tm_min
        start_mins = self._config.mute_start_hour * 60 + self._config.mute_start_minute
        end_mins = self._config.mute_end_hour * 60 + self._config.mute_end_minute
        if start_mins <= end_mins:
            return start_mins <= current_mins < end_mins
        return current_mins >= start_mins or current_mins < end_mins

    def set_cooldown(self, group_id: str, minutes: int) -> None:
        data = self._ensure_group(group_id)
        minutes = max(1, min(120, minutes))
        data.state = GroupState.COOLDOWN
        data.cooldown_until = time.time() + minutes * 60
        data.dirty = True
        self._dirty_groups.add(group_id)

    def add_follow_up(
        self,
        group_id: str,
        user_ids: list[str] | None = None,
    ) -> None:
        data = self._ensure_group(group_id)
        now = time.time()
        ttl = self._config.follow_up_ttl * 60
        if user_ids:
            for uid in user_ids:
                data.follow_up.user_ids.add(uid)
                data.follow_up.user_ttls[uid] = now + ttl
        data.state = GroupState.FOLLOWING
        data.following_since = now
        data.backoff_level = 0
        data.auto_adjust_factor = 1.0
        data.consecutive_replies = 0
        data.dirty = True
        self._dirty_groups.add(group_id)

    def remove_follow_up(
        self,
        group_id: str,
        user_ids: list[str] | None = None,
    ) -> None:
        data = self._ensure_group(group_id)
        if user_ids:
            for uid in user_ids:
                data.follow_up.user_ids.discard(uid)
                data.follow_up.user_ttls.pop(uid, None)
        if not data.follow_up.user_ids:
            data.state = GroupState.IDLE
        data.dirty = True
        self._dirty_groups.add(group_id)

    def _cleanup_expired_follow(self, group_id: str, data: GroupStateData, now: float) -> None:
        expired_users = [uid for uid, ttl in data.follow_up.user_ttls.items() if now >= ttl]
        for uid in expired_users:
            data.follow_up.user_ids.discard(uid)
            data.follow_up.user_ttls.pop(uid, None)
        if expired_users:
            data.dirty = True
            self._dirty_groups.add(group_id)

    def match_follow_up(self, group_id: str, sender_id: str) -> bool:
        data = self.get_state(group_id)
        if data.state == GroupState.COOLDOWN:
            return False
        return sender_id in data.follow_up.user_ids

    def increment_msg_count(self, group_id: str) -> int:
        data = self._ensure_group(group_id)
        data.msg_count += 1
        data.dirty = True
        self._dirty_groups.add(group_id)
        return data.msg_count

    def _current_boost(self, data: GroupStateData) -> float:
        now = time.time()
        if now >= data.boost_until or data.boost_initial >= 1.0:
            return 1.0
        elapsed = now - data.boost_set_at
        total = data.boost_until - data.boost_set_at
        if total <= 0:
            return 1.0
        progress = min(1.0, elapsed / total)
        return data.boost_initial + (1.0 - data.boost_initial) * progress

    def get_effective_thresholds(self, group_id: str) -> tuple[int, int]:
        data = self._ensure_group(group_id)
        backoff_factor = BACKOFF_BASE ** data.backoff_level
        boost = self._current_boost(data)
        combined = data.auto_adjust_factor * backoff_factor * boost
        w_adj = WILLINGNESS_THRESHOLD_ADJUST.get(
            data.willingness, WILLINGNESS_THRESHOLD_ADJUST[DEFAULT_LEVEL]
        )
        effective_n = min(
            max(5, int(self._config.default_n * combined * w_adj["n_factor"])),
            self.N_MAX,
        )
        effective_t = min(
            max(5, int(self._config.default_t * combined * w_adj["t_factor"])),
            self.T_MAX,
        )
        return effective_n, effective_t

    def should_trigger_sampling(self, group_id: str) -> bool:
        data = self.get_state(group_id)
        if data.state == GroupState.COOLDOWN:
            return False
        if self.is_muted():
            return False
        effective_n, effective_t = self.get_effective_thresholds(group_id)
        if data.msg_count >= effective_n:
            return True
        elapsed = time.time() - data.last_sample_time
        if elapsed >= effective_t * 60 and data.msg_count > 0:
            return True
        return False

    def reset_sampling(self, group_id: str) -> None:
        data = self._ensure_group(group_id)
        data.msg_count = 0
        data.last_sample_time = time.time()
        data.dirty = True
        self._dirty_groups.add(group_id)

    def record_skip_reply(self, group_id: str) -> None:
        data = self._ensure_group(group_id)
        if data.backoff_level < MAX_BACKOFF_LEVEL:
            data.backoff_level += 1
        data.consecutive_replies = 0
        data.skip_history.append(time.time())
        self._check_auto_adjust(group_id, data)
        data.dirty = True
        self._dirty_groups.add(group_id)

    def record_actual_reply(self, group_id: str) -> None:
        data = self._ensure_group(group_id)
        data.backoff_level = max(0, data.backoff_level - 1)
        data.consecutive_replies += 1

        max_br = self._config.max_boosted_replies
        if data.consecutive_replies <= max_br:
            strength = self._config.boost_factor
        elif data.consecutive_replies <= max_br * 2:
            over = data.consecutive_replies - max_br
            strength = 1.0 - (1.0 - self._config.boost_factor) * max(0.0, 1.0 - over / max_br)
        else:
            strength = 1.0
            if data.backoff_level < MAX_BACKOFF_LEVEL:
                data.backoff_level = min(data.backoff_level + 1, MAX_BACKOFF_LEVEL)

        now = time.time()
        data.boost_initial = strength
        data.boost_set_at = now
        data.boost_until = now + self._config.boost_duration * 60

        if data.auto_adjust_factor > 1.0:
            data.auto_adjust_factor = max(1.0, data.auto_adjust_factor / 1.2)
        data.dirty = True
        self._dirty_groups.add(group_id)

    def _check_auto_adjust(self, group_id: str, data: GroupStateData) -> None:
        now = time.time()
        one_hour_ago = now - 3600
        recent_skips = sum(1 for t in data.skip_history if t >= one_hour_ago)
        if recent_skips >= 3:
            new_factor = data.auto_adjust_factor * 1.5
            w_adj = WILLINGNESS_THRESHOLD_ADJUST.get(
                data.willingness, WILLINGNESS_THRESHOLD_ADJUST[DEFAULT_LEVEL]
            )
            test_n = int(self._config.default_n * new_factor * w_adj["n_factor"])
            test_t = int(self._config.default_t * new_factor * w_adj["t_factor"])
            if test_n <= self.N_MAX and test_t <= self.T_MAX:
                data.auto_adjust_factor = new_factor

    def reset_group(self, group_id: str) -> None:
        data = self._ensure_group(group_id)
        data.msg_count = 0
        data.last_sample_time = time.time()
        data.backoff_level = 0
        data.auto_adjust_factor = 1.0
        data.consecutive_replies = 0
        data.skip_history.clear()
        data.boost_initial = 1.0
        data.boost_set_at = 0.0
        data.boost_until = 0.0
        data.dirty = True
        self._dirty_groups.add(group_id)

    async def save_dirty(self, save_fn) -> None:
        async with self._global_lock:
            for gid in list(self._dirty_groups):
                data = self._groups.get(gid)
                if data and data.dirty:
                    try:
                        await save_fn(f"state:{gid}", self._serialize_group(data))
                        data.dirty = False
                    except Exception as e:
                        logger.warning(f"Iris Reply: KV save failed for group {gid}: {e}")
            self._dirty_groups.clear()

    def get_willingness(self, group_id: str) -> str:
        data = self._ensure_group(group_id)
        if data.willingness not in VALID_LEVELS:
            data.willingness = DEFAULT_LEVEL
        return data.willingness

    def set_willingness(self, group_id: str, level: str) -> None:
        if level not in VALID_LEVELS:
            return
        data = self._ensure_group(group_id)
        data.willingness = level
        data.dirty = True
        self._dirty_groups.add(group_id)

    def is_whitelisted(self, group_id: str) -> bool:
        return group_id in self._whitelist

    def add_to_whitelist(self, group_id: str) -> None:
        self._whitelist.add(group_id)

    def remove_from_whitelist(self, group_id: str) -> None:
        self._whitelist.discard(group_id)

    def get_whitelist(self) -> set[str]:
        return set(self._whitelist)

    async def load_all(self, load_fn) -> None:
        try:
            wl_data = await load_fn("iris_reply:whitelist")
            if wl_data and isinstance(wl_data, list):
                self._whitelist = set(str(g) for g in wl_data)
        except Exception as e:
            logger.warning(f"Iris Reply: whitelist KV load failed: {e}")
        try:
            all_data = await load_fn("iris_reply:all_groups")
            if not all_data or not isinstance(all_data, dict):
                return
            for gid, gdata in all_data.items():
                self._groups[gid] = self._deserialize_group(gdata)
        except Exception as e:
            logger.warning(f"Iris Reply: KV load failed, running in memory-only mode: {e}")

    async def save_all(self, save_fn) -> None:
        async with self._global_lock:
            try:
                await save_fn("iris_reply:whitelist", list(self._whitelist))
            except Exception as e:
                logger.warning(f"Iris Reply: whitelist KV save failed: {e}")
            all_data = {}
            for gid, data in self._groups.items():
                all_data[gid] = self._serialize_group(data)
                data.dirty = False
            try:
                await save_fn("iris_reply:all_groups", all_data)
            except Exception as e:
                logger.warning(f"Iris Reply: KV save all failed: {e}")
            self._dirty_groups.clear()

    def _serialize_group(self, data: GroupStateData) -> dict[str, Any]:
        return {
            "state": data.state.value,
            "cooldown_until": data.cooldown_until,
            "following_since": data.following_since,
            "msg_count": data.msg_count,
            "last_sample_time": data.last_sample_time,
            "backoff_level": data.backoff_level,
            "auto_adjust_factor": data.auto_adjust_factor,
            "consecutive_replies": data.consecutive_replies,
            "skip_history": list(data.skip_history),
            "follow_up": {
                "user_ids": list(data.follow_up.user_ids),
                "user_ttls": data.follow_up.user_ttls,
            },
            "willingness": data.willingness,
            "boost_initial": data.boost_initial,
            "boost_set_at": data.boost_set_at,
            "boost_until": data.boost_until,
        }

    def _deserialize_group(self, d: dict[str, Any]) -> GroupStateData:
        import math

        follow_up = FollowUpEntry(
            user_ids=set(d.get("follow_up", {}).get("user_ids", [])),
            user_ttls=d.get("follow_up", {}).get("user_ttls", {}),
        )
        willingness = d.get("willingness", DEFAULT_LEVEL)
        if willingness not in VALID_LEVELS:
            willingness = DEFAULT_LEVEL
        if "backoff_level" in d:
            backoff_level = min(d["backoff_level"], MAX_BACKOFF_LEVEL)
        elif "backoff_multiplier" in d:
            old_mult = max(d["backoff_multiplier"], 1)
            backoff_level = min(int(math.log2(old_mult)), MAX_BACKOFF_LEVEL)
        else:
            backoff_level = 0
        return GroupStateData(
            state=GroupState(d.get("state", "idle")),
            cooldown_until=d.get("cooldown_until", 0.0),
            following_since=d.get("following_since", 0.0),
            msg_count=d.get("msg_count", 0),
            last_sample_time=d.get("last_sample_time", time.time()),
            backoff_level=backoff_level,
            auto_adjust_factor=d.get("auto_adjust_factor", 1.0),
            consecutive_replies=d.get("consecutive_replies", 0),
            skip_history=deque(d.get("skip_history", []), maxlen=20),
            follow_up=follow_up,
            willingness=willingness,
            boost_initial=d.get("boost_initial", 1.0),
            boost_set_at=d.get("boost_set_at", 0.0),
            boost_until=d.get("boost_until", 0.0),
        )

    def get_status_text(self, group_id: str) -> str:
        from .prompts import display_level

        data = self.get_state(group_id)
        effective_n, effective_t = self.get_effective_thresholds(group_id)
        backoff_factor = BACKOFF_BASE ** data.backoff_level
        current_boost = self._current_boost(data)
        lines = [
            f"群 {group_id} 状态:",
            f"  状态机: {data.state.value}",
            f"  回复意愿: {display_level(data.willingness)}",
            f"  消息计数: {data.msg_count}/{effective_n}",
            f"  退避等级: {data.backoff_level} (×{backoff_factor:.2f})",
            f"  自动调整倍率: {data.auto_adjust_factor:.1f}",
            f"  频率提升: ×{current_boost:.2f}" + (f" (初始×{data.boost_initial:.2f})" if current_boost < 1.0 else ""),
            f"  连续回复: {data.consecutive_replies}",
            f"  有效阈值: N={effective_n}, T={effective_t}分钟",
        ]
        if data.state == GroupState.COOLDOWN:
            remaining = max(0, data.cooldown_until - time.time())
            lines.append(f"  冷却剩余: {remaining / 60:.1f} 分钟")
        if current_boost < 1.0:
            remaining = max(0, data.boost_until - time.time())
            lines.append(f"  Boost 剩余: {remaining / 60:.1f} 分钟")
        if data.follow_up.user_ids:
            lines.append(f"  跟进用户: {', '.join(data.follow_up.user_ids)}")
        return "\n".join(lines)
