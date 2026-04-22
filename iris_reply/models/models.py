from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ReplyType(Enum):
    PROACTIVE = "proactive"
    FOLLOWUP = "followup"
    KEYWORD = "keyword"


class FollowupType(Enum):
    TOPIC_EXTEND = "topic_extend"
    UNFINISHED = "unfinished"
    EMOTIONAL_CARE = "emotional_care"
    CONFIRMATION = "confirmation"


class KeywordSource(Enum):
    STATIC = "static"
    PROFILE = "profile"
    CONVERSATION = "conversation"
    KNOWLEDGE = "knowledge"


class KeywordMatchType(Enum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"


@dataclass
class AnalysisResult:
    should_consider: bool = False
    topic_relevance: float = 0.0
    intent_signal: float = 0.0
    emotion_signal: float = 0.0
    silence_gap: float = 0.0
    mention_signal: float = 0.0
    summary: str = ""


@dataclass
class Decision:
    should_reply: bool = False
    confidence: float = 0.0
    reason: str = ""
    reply_content: str | None = None
    reply_direction: str | None = None


@dataclass
class AssembledContext:
    recent_messages: list[dict[str, Any]] = field(default_factory=list)
    relevant_memories: list[dict[str, Any]] = field(default_factory=list)
    knowledge_facts: list[dict[str, Any]] = field(default_factory=list)
    user_profile: dict[str, Any] | None = None
    group_profile: dict[str, Any] | None = None


@dataclass
class KeywordMatch:
    keyword: str = ""
    match_type: KeywordMatchType = KeywordMatchType.EXACT
    confidence: float = 1.0
    position: int = -1
    source: KeywordSource = KeywordSource.STATIC


@dataclass
class ValidationResult:
    should_reply: bool = False
    confidence: float = 0.0
    reason: str = ""
    reply_direction: str | None = None


@dataclass
class FollowupPlan:
    should_followup: bool = False
    followup_type: FollowupType = FollowupType.TOPIC_EXTEND
    delay_seconds: int = 60
    direction: str = ""
    max_wait_messages: int = 5


@dataclass
class GroupConfig:
    group_id: str = ""
    proactive_enabled: bool = False
    followup_enabled: bool = False
    cooldown_seconds: int = 300
