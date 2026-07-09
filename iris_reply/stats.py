from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMCallLog:
    group_id: str
    trigger_reason: str
    system_prompt: str
    user_prompt: str
    response_text: str
    should_reply: bool
    observation: str
    follow_up_users: list[str]
    follow_up_keywords: list[str]
    interest_reason: str
    topic_drifted: bool
    timestamp: float
    duration_ms: float = 0.0


@dataclass
class GroupStats:
    group_id: str
    total_triggers: int = 0
    total_replies: int = 0
    total_skips: int = 0
    total_errors: int = 0
    total_drifts: int = 0
    total_passive_replies: int = 0
    last_trigger_time: float = 0.0
    last_trigger_reason: str = ""
    last_reply_time: float = 0.0
    current_state: str = "idle"
    willingness: str = "medium"
    msg_count: int = 0
    effective_n: int = 0
    effective_t: int = 0
    backoff_level: int = 0
    consecutive_replies: int = 0


MAX_LOG_ENTRIES = 500


class StatsCollector:
    def __init__(self) -> None:
        self._enabled: bool = False
        self._llm_logs: deque[LLMCallLog] = deque(maxlen=MAX_LOG_ENTRIES)
        self._group_stats: dict[str, GroupStats] = {}
        self._pending_call: dict[str, dict[str, Any]] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def _ensure_group(self, group_id: str) -> GroupStats:
        if group_id not in self._group_stats:
            self._group_stats[group_id] = GroupStats(group_id=group_id)
        return self._group_stats[group_id]

    def record_trigger_start(
        self,
        group_id: str,
        trigger_reason: str,
        system_prompt: str,
        user_prompt: str,
    ) -> None:
        if not self._enabled:
            return
        self._pending_call[group_id] = {
            "trigger_reason": trigger_reason,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "start_time": time.time(),
        }
        gs = self._ensure_group(group_id)
        gs.total_triggers += 1
        gs.last_trigger_time = time.time()
        gs.last_trigger_reason = trigger_reason

    def record_trigger_result(
        self,
        group_id: str,
        response_text: str,
        should_reply: bool,
        observation: str,
        follow_up_users: list[str],
        follow_up_keywords: list[str],
        interest_reason: str,
        topic_drifted: bool,
    ) -> None:
        if not self._enabled:
            return
        pending = self._pending_call.pop(group_id, None)
        if not pending:
            return
        start_time = pending["start_time"]
        duration_ms = (time.time() - start_time) * 1000

        log = LLMCallLog(
            group_id=group_id,
            trigger_reason=pending["trigger_reason"],
            system_prompt=pending["system_prompt"],
            user_prompt=pending["user_prompt"],
            response_text=response_text,
            should_reply=should_reply,
            observation=observation,
            follow_up_users=follow_up_users,
            follow_up_keywords=follow_up_keywords,
            interest_reason=interest_reason,
            topic_drifted=topic_drifted,
            timestamp=time.time(),
            duration_ms=duration_ms,
        )
        self._llm_logs.append(log)

        gs = self._ensure_group(group_id)
        if topic_drifted:
            gs.total_drifts += 1
        elif should_reply:
            gs.total_replies += 1
            gs.last_reply_time = time.time()
        else:
            gs.total_skips += 1

    def record_trigger_error(self, group_id: str) -> None:
        if not self._enabled:
            return
        self._pending_call.pop(group_id, None)
        gs = self._ensure_group(group_id)
        gs.total_errors += 1

    def record_passive_reply(self, group_id: str) -> None:
        if not self._enabled:
            return
        gs = self._ensure_group(group_id)
        gs.total_passive_replies += 1
        gs.last_reply_time = time.time()

    def update_group_state(
        self,
        group_id: str,
        state: str,
        willingness: str,
        msg_count: int,
        effective_n: int,
        effective_t: int,
        backoff_level: int,
        consecutive_replies: int,
    ) -> None:
        if not self._enabled:
            return
        gs = self._ensure_group(group_id)
        gs.current_state = state
        gs.willingness = willingness
        gs.msg_count = msg_count
        gs.effective_n = effective_n
        gs.effective_t = effective_t
        gs.backoff_level = backoff_level
        gs.consecutive_replies = consecutive_replies

    def get_group_summaries(self) -> list[dict[str, Any]]:
        result = []
        for gs in self._group_stats.values():
            result.append({
                "group_id": gs.group_id,
                "total_triggers": gs.total_triggers,
                "total_replies": gs.total_replies,
                "total_skips": gs.total_skips,
                "total_errors": gs.total_errors,
                "total_drifts": gs.total_drifts,
                "total_passive_replies": gs.total_passive_replies,
                "last_trigger_time": gs.last_trigger_time,
                "last_trigger_reason": gs.last_trigger_reason,
                "last_reply_time": gs.last_reply_time,
                "current_state": gs.current_state,
                "willingness": gs.willingness,
                "msg_count": gs.msg_count,
                "effective_n": gs.effective_n,
                "effective_t": gs.effective_t,
                "backoff_level": gs.backoff_level,
                "consecutive_replies": gs.consecutive_replies,
            })
        result.sort(key=lambda x: x["last_trigger_time"], reverse=True)
        return result

    def get_llm_logs(
        self,
        group_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        logs = list(self._llm_logs)
        if group_id:
            logs = [l for l in logs if l.group_id == group_id]
        logs = logs[::-1]
        sliced = logs[offset:offset + limit]
        result = []
        for log in sliced:
            result.append({
                "group_id": log.group_id,
                "trigger_reason": log.trigger_reason,
                "system_prompt": log.system_prompt,
                "user_prompt": log.user_prompt,
                "response_text": log.response_text,
                "should_reply": log.should_reply,
                "observation": log.observation,
                "follow_up_users": log.follow_up_users,
                "follow_up_keywords": log.follow_up_keywords,
                "interest_reason": log.interest_reason,
                "topic_drifted": log.topic_drifted,
                "timestamp": log.timestamp,
                "duration_ms": round(log.duration_ms, 1),
            })
        return result

    def get_group_detail(self, group_id: str) -> dict[str, Any] | None:
        gs = self._group_stats.get(group_id)
        if not gs:
            return None
        return {
            "group_id": gs.group_id,
            "total_triggers": gs.total_triggers,
            "total_replies": gs.total_replies,
            "total_skips": gs.total_skips,
            "total_errors": gs.total_errors,
            "total_drifts": gs.total_drifts,
            "total_passive_replies": gs.total_passive_replies,
            "last_trigger_time": gs.last_trigger_time,
            "last_trigger_reason": gs.last_trigger_reason,
            "last_reply_time": gs.last_reply_time,
            "current_state": gs.current_state,
            "willingness": gs.willingness,
            "msg_count": gs.msg_count,
            "effective_n": gs.effective_n,
            "effective_t": gs.effective_t,
            "backoff_level": gs.backoff_level,
            "consecutive_replies": gs.consecutive_replies,
        }

    def clear_logs(self) -> None:
        self._llm_logs.clear()
        self._group_stats.clear()
        self._pending_call.clear()
