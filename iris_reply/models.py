from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class SignalType(str, Enum):
    EMOTION_HIGH = "emotion_high"
    RULE_MATCH = "rule_match"


class FollowUpReplyType(str, Enum):
    ACKNOWLEDGE = "acknowledge"
    CONTINUE_TOPIC = "continue_topic"
    EMOTION_SUPPORT = "emotion_support"
    QUESTION = "question"


@dataclass
class Signal:
    signal_type: SignalType
    session_key: str
    group_id: str
    user_id: str
    weight: float
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    signal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at


@dataclass
class FollowUpExpectation:
    session_key: str
    group_id: str
    trigger_user_id: str
    trigger_message: str
    bot_reply_summary: str
    followup_window_end: datetime
    short_window_end: Optional[datetime] = None
    aggregated_messages: List[Dict[str, Any]] = field(default_factory=list)
    followup_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    recent_context: List[Dict[str, Any]] = field(default_factory=list)
    expectation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    _processing: bool = field(default=False, repr=False)

    @property
    def is_window_expired(self) -> bool:
        return datetime.now() >= self.followup_window_end

    @property
    def is_short_window_expired(self) -> bool:
        if self.short_window_end is None:
            return False
        return datetime.now() >= self.short_window_end

    @property
    def has_aggregated_messages(self) -> bool:
        return len(self.aggregated_messages) > 0


@dataclass
class AggregatedDecision:
    should_reply: bool
    session_key: str
    group_id: str
    target_user_id: str = ""
    aggregated_weight: float = 0.0
    signals: List[Signal] = field(default_factory=list)
    reason: str = ""
    recent_messages: List[Dict[str, Any]] = field(default_factory=list)
    llm_confirmed: bool = False


@dataclass
class FollowUpDecision:
    should_reply: bool
    reason: str = ""
    reply_type: FollowUpReplyType = FollowUpReplyType.ACKNOWLEDGE
    suggested_direction: str = ""


@dataclass
class ProactiveReplyResult:
    trigger_prompt: str
    reply_params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    group_id: str = ""
    session_key: str = ""
    target_user: str = ""
    recent_messages: List[Dict[str, Any]] = field(default_factory=list)
    emotion_summary: str = ""
    source: str = "signal_queue"


class ReplyDecisionType(str, Enum):
    SKIP = "skip"
    NO_SIGNAL = "no_signal"
    SIGNAL_ENQUEUED = "signal_enqueued"
    FOLLOWUP_AGGREGATED = "followup_aggregated"


@dataclass
class ReplyDecision:
    decision_type: ReplyDecisionType
    reason: str = ""


@dataclass(frozen=True)
class CooldownState:
    group_id: str
    started_at: datetime
    expires_at: datetime
    initiated_by: str
    reason: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return datetime.now(timezone.utc) < self.expires_at

    @property
    def remaining_seconds(self) -> int:
        delta = self.expires_at - datetime.now(timezone.utc)
        return max(0, int(delta.total_seconds()))

    @property
    def remaining_minutes(self) -> int:
        import math
        return math.ceil(self.remaining_seconds / 60)

    @property
    def duration_minutes(self) -> int:
        delta = self.expires_at - self.started_at
        return int(delta.total_seconds() / 60)

    def format_remaining(self) -> str:
        total = self.remaining_seconds
        minutes, seconds = divmod(total, 60)
        return f"{minutes}分{seconds:02d}秒"

    def format_expires_at_local(self) -> str:
        local_time = self.expires_at.astimezone()
        return local_time.strftime("%H:%M")


@dataclass
class QuietHoursConfig:
    start_hour: int
    end_hour: int

    def is_active_now(self) -> bool:
        hour = datetime.now().hour
        if self.start_hour <= self.end_hour:
            return self.start_hour <= hour < self.end_hour
        return hour >= self.start_hour or hour < self.end_hour

    def to_dict(self) -> Dict[str, Any]:
        return {"start_hour": self.start_hour, "end_hour": self.end_hour}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QuietHoursConfig:
        return cls(
            start_hour=int(data["start_hour"]),
            end_hour=int(data["end_hour"]),
        )
