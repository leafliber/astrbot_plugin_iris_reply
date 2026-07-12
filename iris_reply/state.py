from __future__ import annotations

import asyncio
import math
import time
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
    keywords: set[str] = field(default_factory=set)
    keyword_ttls: dict[str, float] = field(default_factory=dict)
    reason: str = ""


@dataclass
class GroupStateData:
    state: GroupState = GroupState.IDLE
    cooldown_until: float = 0.0
    following_since: float = 0.0
    msg_count: int = 0
    last_sample_time: float = 0.0
    backoff_level: int = 0
    last_backoff_time: float = 0.0
    consecutive_replies: int = 0
    follow_up: FollowUpEntry = field(default_factory=FollowUpEntry)
    willingness: str = DEFAULT_LEVEL
    boost_initial: float = 1.0
    boost_set_at: float = 0.0
    boost_until: float = 0.0
    last_detect_time: float = 0.0
    dirty: bool = False


class StateManager:
    N_MAX = 120
    T_MAX = 300
    MAX_FOLLOW_UP_USERS = 20
    MAX_FOLLOW_UP_KEYWORDS = 10
    BACKOFF_DECAY_INTERVAL = 300.0

    _GROUP_IDS_KEY = "iris_reply:group_ids"
    _GROUP_KEY_PREFIX = "state:"
    _LEGACY_BULK_KEY = "iris_reply:all_groups"

    def __init__(self, config: ConfigManager) -> None:
        self._config = config
        self._groups: dict[str, GroupStateData] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._dirty_groups: set[str] = set()
        self._global_lock = asyncio.Lock()
        self._whitelist: set[str] = set()
        self._whitelist_dirty: bool = False
        self._group_ids_dirty: bool = False

    def get_lock(self, group_id: str) -> asyncio.Lock:
        return self._locks.setdefault(group_id, asyncio.Lock())

    def remove_group_lock(self, group_id: str) -> None:
        self._locks.pop(group_id, None)

    def _mark_dirty(self, group_id: str, data: GroupStateData) -> None:
        data.dirty = True
        self._dirty_groups.add(group_id)

    def _ensure_group(self, group_id: str) -> GroupStateData:
        if group_id not in self._groups:
            self._groups[group_id] = GroupStateData(
                last_sample_time=time.time(),
            )
            self._group_ids_dirty = True
        return self._groups[group_id]

    def get_state(self, group_id: str) -> GroupStateData:
        data = self._ensure_group(group_id)
        now = time.time()
        if data.state == GroupState.COOLDOWN and now >= data.cooldown_until:
            data.state = GroupState.IDLE
            self._mark_dirty(group_id, data)
        if data.state == GroupState.FOLLOWING:
            self._cleanup_expired_follow(group_id, data, now)
            if not data.follow_up.user_ids and not data.follow_up.keywords:
                data.state = GroupState.IDLE
                self._mark_dirty(group_id, data)
        return data

    def is_muted(self) -> bool:
        now = time.localtime()
        current_mins = now.tm_hour * 60 + now.tm_min
        start_hour, start_minute, end_hour, end_minute = self._config.mute_period
        start_mins = start_hour * 60 + start_minute
        end_mins = end_hour * 60 + end_minute
        if start_mins <= end_mins:
            return start_mins <= current_mins < end_mins
        return current_mins >= start_mins or current_mins < end_mins

    def set_cooldown(self, group_id: str, minutes: int) -> int:
        data = self._ensure_group(group_id)
        minutes = max(1, min(self.N_MAX, minutes))
        data.state = GroupState.COOLDOWN
        data.cooldown_until = time.time() + minutes * 60
        self._mark_dirty(group_id, data)
        return minutes

    def add_follow_up(
        self,
        group_id: str,
        user_ids: list[str] | None = None,
        keywords: list[str] | None = None,
        reason: str = "",
        ttl_minutes: float | None = None,
    ) -> None:
        data = self._ensure_group(group_id)
        now = time.time()
        ttl = (ttl_minutes if ttl_minutes is not None else self._config.follow_up_ttl) * 60
        was_following = data.state == GroupState.FOLLOWING
        if user_ids:
            for uid in user_ids:
                if len(data.follow_up.user_ids) >= self.MAX_FOLLOW_UP_USERS:
                    break
                data.follow_up.user_ids.add(uid)
                data.follow_up.user_ttls[uid] = now + ttl
        if keywords:
            for kw in keywords:
                if len(data.follow_up.keywords) >= self.MAX_FOLLOW_UP_KEYWORDS:
                    break
                data.follow_up.keywords.add(kw)
                data.follow_up.keyword_ttls[kw] = now + ttl
        if reason:
            data.follow_up.reason = reason
        if not was_following:
            data.state = GroupState.FOLLOWING
            data.following_since = now
        self._mark_dirty(group_id, data)

    def remove_follow_up(
        self,
        group_id: str,
        user_ids: list[str] | None = None,
        keywords: list[str] | None = None,
    ) -> None:
        data = self._ensure_group(group_id)
        if user_ids:
            for uid in user_ids:
                data.follow_up.user_ids.discard(uid)
                data.follow_up.user_ttls.pop(uid, None)
        if keywords:
            for kw in keywords:
                data.follow_up.keywords.discard(kw)
                data.follow_up.keyword_ttls.pop(kw, None)
        if not data.follow_up.user_ids and not data.follow_up.keywords:
            data.state = GroupState.IDLE
        self._mark_dirty(group_id, data)

    def _cleanup_expired_follow(self, group_id: str, data: GroupStateData, now: float) -> None:
        expired_users = [uid for uid, ttl in data.follow_up.user_ttls.items() if now >= ttl]
        for uid in expired_users:
            data.follow_up.user_ids.discard(uid)
            data.follow_up.user_ttls.pop(uid, None)
        expired_keywords = [kw for kw, ttl in data.follow_up.keyword_ttls.items() if now >= ttl]
        for kw in expired_keywords:
            data.follow_up.keywords.discard(kw)
            data.follow_up.keyword_ttls.pop(kw, None)
        if expired_users or expired_keywords:
            self._mark_dirty(group_id, data)

    def match_follow_up(self, group_id: str, sender_id: str) -> bool:
        data = self.get_state(group_id)
        if data.state == GroupState.COOLDOWN:
            return False
        return sender_id in data.follow_up.user_ids

    def match_keyword(self, group_id: str, text: str) -> list[str]:
        data = self.get_state(group_id)
        if data.state == GroupState.COOLDOWN:
            return []
        matched = []
        text_lower = text.lower()
        for kw in data.follow_up.keywords:
            if kw.lower() in text_lower:
                matched.append(kw)
        return matched

    def get_follow_up_info(self, group_id: str) -> tuple[list[str], list[str], str]:
        data = self.get_state(group_id)
        return sorted(data.follow_up.user_ids), sorted(data.follow_up.keywords), data.follow_up.reason

    def increment_msg_count(self, group_id: str) -> int:
        data = self._ensure_group(group_id)
        data.msg_count += 1
        self._mark_dirty(group_id, data)
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

    def _decay_backoff(self, group_id: str, data: GroupStateData, now: float) -> None:
        if data.backoff_level <= 0 or data.last_backoff_time <= 0:
            return
        elapsed = now - data.last_backoff_time
        levels_to_decay = int(elapsed / self.BACKOFF_DECAY_INTERVAL)
        if levels_to_decay > 0:
            data.backoff_level = max(0, data.backoff_level - levels_to_decay)
            data.last_backoff_time = now
            self._mark_dirty(group_id, data)

    def get_effective_thresholds(self, group_id: str) -> tuple[int, int]:
        data = self._ensure_group(group_id)
        now = time.time()
        self._decay_backoff(group_id, data, now)
        backoff_factor = BACKOFF_BASE ** data.backoff_level
        boost = self._current_boost(data)
        combined = backoff_factor * boost
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
        self._mark_dirty(group_id, data)

    def record_skip_reply(self, group_id: str) -> None:
        data = self._ensure_group(group_id)
        if data.backoff_level < MAX_BACKOFF_LEVEL:
            data.backoff_level += 1
        if data.last_backoff_time <= 0:
            data.last_backoff_time = time.time()
        data.consecutive_replies = 0
        self._mark_dirty(group_id, data)

    def record_actual_reply(self, group_id: str, *, count_consecutive: bool = True) -> None:
        data = self._ensure_group(group_id)
        if data.consecutive_replies == 0:
            data.backoff_level = max(0, data.backoff_level - 1)
        if count_consecutive:
            data.consecutive_replies += 1

        max_br = self._config.max_boosted_replies
        now = time.time()

        if data.consecutive_replies <= max_br:
            data.boost_initial = self._config.boost_factor
            data.boost_set_at = now
            data.boost_until = now + self._config.boost_duration * 60
        else:
            data.boost_initial = 1.0
            fatigue = min(data.consecutive_replies - max_br, MAX_BACKOFF_LEVEL)
            data.backoff_level = max(data.backoff_level, fatigue)
            data.last_backoff_time = now

        self._mark_dirty(group_id, data)

    def can_detect(self, group_id: str, *, follow_up: bool = False) -> bool:
        data = self._ensure_group(group_id)
        now = time.time()
        min_interval = float(self._config.trigger_min_interval)
        if follow_up:
            min_interval *= 0.5
        return (now - data.last_detect_time) >= min_interval

    def record_detect_time(self, group_id: str) -> None:
        data = self._ensure_group(group_id)
        data.last_detect_time = time.time()

    def clear_follow_up(self, group_id: str) -> None:
        data = self._ensure_group(group_id)
        data.follow_up.user_ids.clear()
        data.follow_up.user_ttls.clear()
        data.follow_up.keywords.clear()
        data.follow_up.keyword_ttls.clear()
        data.follow_up.reason = ""
        data.state = GroupState.IDLE
        self._mark_dirty(group_id, data)

    def reset_group(self, group_id: str) -> None:
        data = self._ensure_group(group_id)
        data.msg_count = 0
        data.last_sample_time = time.time()
        data.backoff_level = 0
        data.last_backoff_time = 0.0
        data.consecutive_replies = 0
        data.boost_initial = 1.0
        data.boost_set_at = 0.0
        data.boost_until = 0.0
        data.last_detect_time = 0.0
        data.follow_up.user_ids.clear()
        data.follow_up.user_ttls.clear()
        data.follow_up.keywords.clear()
        data.follow_up.keyword_ttls.clear()
        data.follow_up.reason = ""
        data.following_since = 0.0
        data.state = GroupState.IDLE
        self._mark_dirty(group_id, data)

    async def save_dirty(self, save_fn) -> None:
        async with self._global_lock:
            if self._whitelist_dirty:
                self._whitelist_dirty = False
                try:
                    await save_fn("iris_reply:whitelist", list(self._whitelist))
                except Exception as e:
                    self._whitelist_dirty = True
                    logger.warning("Iris Reply: whitelist KV save failed: %s", e)
            if self._group_ids_dirty:
                self._group_ids_dirty = False
                try:
                    await save_fn(self._GROUP_IDS_KEY, list(self._groups.keys()))
                except Exception as e:
                    self._group_ids_dirty = True
                    logger.warning("Iris Reply: group manifest KV save failed: %s", e)
            dirty_snapshot = list(self._dirty_groups)
            self._dirty_groups.clear()
            for gid in dirty_snapshot:
                data = self._groups.get(gid)
                if data:
                    data.dirty = False
            for gid in dirty_snapshot:
                data = self._groups.get(gid)
                if not data:
                    continue
                snapshot = self._serialize_group(data)
                try:
                    await save_fn(f"{self._GROUP_KEY_PREFIX}{gid}", snapshot)
                except Exception as e:
                    data.dirty = True
                    self._dirty_groups.add(gid)
                    logger.warning("Iris Reply: KV save failed for group %s: %s", gid, e)

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
        self._mark_dirty(group_id, data)

    def is_whitelisted(self, group_id: str) -> bool:
        return group_id in self._whitelist

    def add_to_whitelist(self, group_id: str) -> None:
        self._whitelist.add(group_id)
        self._whitelist_dirty = True

    def remove_from_whitelist(self, group_id: str) -> None:
        self._whitelist.discard(group_id)
        self._whitelist_dirty = True

    def get_whitelist(self) -> set[str]:
        return set(self._whitelist)

    async def load_all(self, load_fn) -> None:
        try:
            wl_data = await load_fn("iris_reply:whitelist")
            if wl_data and isinstance(wl_data, list):
                self._whitelist = set(str(g) for g in wl_data)
        except Exception as e:
            logger.warning("Iris Reply: whitelist KV load failed: %s", e)

        loaded_any = False
        try:
            group_ids = await load_fn(self._GROUP_IDS_KEY)
            if group_ids and isinstance(group_ids, list):
                for gid in group_ids:
                    gdata = await load_fn(f"{self._GROUP_KEY_PREFIX}{str(gid)}")
                    if gdata and isinstance(gdata, dict):
                        self._groups[str(gid)] = self._deserialize_group(gdata)
                        loaded_any = True
        except Exception as e:
            logger.warning("Iris Reply: per-group KV load failed: %s", e)

        if not loaded_any:
            try:
                all_data = await load_fn(self._LEGACY_BULK_KEY)
                if all_data and isinstance(all_data, dict):
                    for gid, gdata in all_data.items():
                        self._groups[str(gid)] = self._deserialize_group(gdata)
                    logger.info("Iris Reply: loaded %d groups from legacy bulk format", len(all_data))
            except Exception as e:
                logger.warning("Iris Reply: KV load failed, running in memory-only mode: %s", e)

    async def save_all(self, save_fn) -> None:
        async with self._global_lock:
            try:
                await save_fn("iris_reply:whitelist", list(self._whitelist))
                self._whitelist_dirty = False
            except Exception as e:
                logger.warning("Iris Reply: whitelist KV save failed: %s", e)
            all_gids = list(self._groups.keys())
            self._dirty_groups.clear()
            for gid, data in self._groups.items():
                data.dirty = False
            for gid in all_gids:
                data = self._groups.get(gid)
                if not data:
                    continue
                snapshot = self._serialize_group(data)
                try:
                    await save_fn(f"{self._GROUP_KEY_PREFIX}{gid}", snapshot)
                except Exception as e:
                    data.dirty = True
                    self._dirty_groups.add(gid)
                    logger.warning("Iris Reply: KV save failed for group %s: %s", gid, e)
            try:
                await save_fn(self._GROUP_IDS_KEY, list(self._groups.keys()))
                self._group_ids_dirty = False
            except Exception as e:
                self._group_ids_dirty = True
                logger.warning("Iris Reply: group manifest KV save failed: %s", e)

    def _serialize_group(self, data: GroupStateData) -> dict[str, Any]:
        return {
            "state": data.state.value,
            "cooldown_until": data.cooldown_until,
            "following_since": data.following_since,
            "msg_count": data.msg_count,
            "last_sample_time": data.last_sample_time,
            "backoff_level": data.backoff_level,
            "last_backoff_time": data.last_backoff_time,
            "consecutive_replies": data.consecutive_replies,
            "follow_up": {
                "user_ids": list(data.follow_up.user_ids),
                "user_ttls": data.follow_up.user_ttls,
                "keywords": list(data.follow_up.keywords),
                "keyword_ttls": data.follow_up.keyword_ttls,
                "reason": data.follow_up.reason,
            },
            "willingness": data.willingness,
            "boost_initial": data.boost_initial,
            "boost_set_at": data.boost_set_at,
            "boost_until": data.boost_until,
            "last_detect_time": data.last_detect_time,
        }

    def _deserialize_group(self, d: dict[str, Any]) -> GroupStateData:
        follow_up = FollowUpEntry(
            user_ids=set(d.get("follow_up", {}).get("user_ids", [])),
            user_ttls=d.get("follow_up", {}).get("user_ttls", {}),
            keywords=set(d.get("follow_up", {}).get("keywords", [])),
            keyword_ttls=d.get("follow_up", {}).get("keyword_ttls", {}),
            reason=d.get("follow_up", {}).get("reason", ""),
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
        last_backoff_time = d.get("last_backoff_time", 0.0)
        if backoff_level > 0 and last_backoff_time <= 0:
            last_backoff_time = time.time()
        return GroupStateData(
            state=GroupState(d.get("state", "idle")),
            cooldown_until=d.get("cooldown_until", 0.0),
            following_since=d.get("following_since", 0.0),
            msg_count=d.get("msg_count", 0),
            last_sample_time=d.get("last_sample_time", time.time()),
            backoff_level=backoff_level,
            last_backoff_time=last_backoff_time,
            consecutive_replies=d.get("consecutive_replies", 0),
            follow_up=follow_up,
            willingness=willingness,
            boost_initial=d.get("boost_initial", 1.0),
            boost_set_at=d.get("boost_set_at", 0.0),
            boost_until=d.get("boost_until", 0.0),
            last_detect_time=d.get("last_detect_time", 0.0),
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
            lines.append(f"  跟进用户: {', '.join(sorted(data.follow_up.user_ids))}")
        if data.follow_up.keywords:
            lines.append(f"  跟进关键词: {', '.join(sorted(data.follow_up.keywords))}")
        if data.follow_up.reason:
            lines.append(f"  跟进原因: {data.follow_up.reason}")
        return "\n".join(lines)
