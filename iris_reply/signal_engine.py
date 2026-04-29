from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from astrbot.api import logger

from .models import Signal, SignalType

QUESTION_KEYWORDS: List[str] = [
    "吗", "呢", "什么", "怎么", "为什么", "如何", "哪里", "哪个",
    "几个", "几点", "多少", "能不能", "可以吗", "是不是", "有没有",
    "谁", "怎样", "何时", "咱", "啊", "哪样",
    "how", "what", "why", "where", "when",
]

MENTION_PATTERNS: List[str] = [
    "你说", "你怎么看", "你觉得", "你认为",
    "帮我", "帮忙", "求助",
]

EMOTION_POSITIVE: List[str] = [
    "开心", "高兴", "太好了", "成功了", "庆祝", "激动", "兴奋",
    "棒", "厉害", "牛", "绝了", "爽",
    "绝绝子", "yyds", "赞", "欧耶", "嘴角上扬",
    "开花", "小确幸", "笑死",
]

EMOTION_NEGATIVE: List[str] = [
    "难过", "伤心", "烦", "累", "焦虑", "压力", "失眠", "崩溃",
    "无聊", "不爽", "郁闷", "痛苦", "绝望", "迷茫",
    "破防", "emo", "裂开", "麻了", "不想努力了",
    "不开心", "心累", "自闭",
]

ATTENTION_KEYWORDS: List[str] = [
    "有人吗", "在吗", "出来聊天", "好无聊", "陪我", "说说话",
    "有没有人", "谁在", "好寂寞", "一个人好无聊",
]

SHORT_CONFIRM_PATTERNS: List[str] = [
    "嗯", "哦", "好的", "好吧", "行", "ok", "OK", "Ok",
    "收到", "了解", "知道了", "明白",
]

EMOJI_ONLY_PATTERN = re.compile(
    r"^(?:"
    r"[\s"
    r"\U0001F600-\U0001F64F"
    r"\U0001F300-\U0001F5FF"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF"
    r"\U0001F900-\U0001F9FF"
    r"\U0001FA00-\U0001FA6F"
    r"\U0001FA70-\U0001FAFF"
    r"\U00002702-\U000027B0"
    r"\U00002600-\U000026FF"
    r"\U0000FE00-\U0000FE0F"
    r"\U0000200D"
    r"\U000023E9-\U000023F3"
    r"\U000023F8-\U000023FA"
    r"]"
    r"|\[\w+\]"
    r")+$"
)


class SignalGenerator:
    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._question_keywords = config.get("question_keywords") or QUESTION_KEYWORDS
        self._mention_keywords = config.get("mention_keywords") or MENTION_PATTERNS
        self._attention_keywords = config.get("attention_keywords") or ATTENTION_KEYWORDS
        self._emotion_keywords = config.get("emotion_keywords") or (EMOTION_NEGATIVE + EMOTION_POSITIVE)

    def generate(
        self,
        text: str,
        user_id: str,
        group_id: str,
        session_key: str,
        emotion_intensity: float = 0.0,
    ) -> List[Signal]:
        signals: List[Signal] = []

        if not text or not text.strip():
            return signals

        text = text.strip()

        if self._is_short_confirm(text) or self._is_emoji_only(text):
            return signals

        rule_signal = self._detect_rule_match(text, user_id, group_id, session_key)
        if rule_signal:
            signals.append(rule_signal)

        emotion_signal = self._detect_emotion_high(
            text, user_id, group_id, session_key, emotion_intensity
        )
        if emotion_signal:
            signals.append(emotion_signal)

        return signals

    def _detect_rule_match(
        self,
        text: str,
        user_id: str,
        group_id: str,
        session_key: str,
    ) -> Optional[Signal]:
        score = 0.0
        matched: List[str] = []

        q_score = self._detect_question(text)
        if q_score > 0:
            score += q_score
            matched.append("question")

        mention_score = self._detect_mention(text)
        if mention_score > 0:
            score += mention_score
            matched.append("mention")

        attention_score = self._detect_attention(text)
        if attention_score > 0:
            score += attention_score
            matched.append("attention")

        emo_score, emo_type = self._detect_emotion_keywords(text)
        if emo_score > 0:
            score += emo_score
            matched.append(f"emotion_{emo_type}")

        score = max(0.0, min(1.0, score))

        if score < 0.2:
            return None

        ttl = self._config.get("signal_ttl_rule_match", 300)
        return Signal(
            signal_type=SignalType.RULE_MATCH,
            session_key=session_key,
            group_id=group_id,
            user_id=user_id,
            weight=score,
            expires_at=datetime.now() + timedelta(seconds=ttl),
            metadata={"matched_rules": matched, "text_preview": text[:50]},
        )

    def _detect_emotion_high(
        self,
        text: str,
        user_id: str,
        group_id: str,
        session_key: str,
        emotion_intensity: float,
    ) -> Optional[Signal]:
        if emotion_intensity < 0.7:
            return None

        ttl = self._config.get("signal_ttl_emotion_high", 180)
        weight = min(1.0, 0.7 + (emotion_intensity - 0.7) * 1.0)

        return Signal(
            signal_type=SignalType.EMOTION_HIGH,
            session_key=session_key,
            group_id=group_id,
            user_id=user_id,
            weight=weight,
            expires_at=datetime.now() + timedelta(seconds=ttl),
            metadata={
                "emotion_intensity": emotion_intensity,
                "text_preview": text[:50],
            },
        )

    def _detect_question(self, text: str) -> float:
        count = sum(1 for kw in self._question_keywords if kw in text)
        if count == 0:
            return 0.0
        if text.rstrip().endswith("?") or text.rstrip().endswith("？"):
            return 0.3
        return min(0.3, count * 0.15)

    def _detect_mention(self, text: str) -> float:
        for pattern in self._mention_keywords:
            if pattern in text:
                return 0.4
        return 0.0

    def _detect_attention(self, text: str) -> float:
        for kw in self._attention_keywords:
            if kw in text:
                return 0.2
        return 0.0

    def _detect_emotion_keywords(self, text: str) -> Tuple[float, str]:
        neg_count = sum(1 for kw in EMOTION_NEGATIVE if kw in text)
        pos_count = sum(1 for kw in EMOTION_POSITIVE if kw in text)

        if neg_count > pos_count:
            return min(0.25, neg_count * 0.1), "negative"
        elif pos_count > 0:
            return min(0.15, pos_count * 0.08), "positive"
        return 0.0, ""

    @staticmethod
    def _is_short_confirm(text: str) -> bool:
        stripped = text.strip().rstrip("。.!！~")
        return stripped in SHORT_CONFIRM_PATTERNS

    @staticmethod
    def _is_emoji_only(text: str) -> bool:
        if not text.strip():
            return True
        return EMOJI_ONLY_PATTERN.match(text.strip()) is not None


class SignalQueue:
    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._queues: Dict[str, List[Signal]] = {}
        self._last_message_time: Dict[str, datetime] = {}

    def enqueue(self, signal: Signal) -> bool:
        group_id = signal.group_id
        if group_id not in self._queues:
            self._queues[group_id] = []

        max_signals = self._config.get("signal_max_signals_per_group", 50)
        if len(self._queues[group_id]) >= max_signals:
            min_idx = min(
                range(len(self._queues[group_id])),
                key=lambda i: self._queues[group_id][i].weight,
            )
            removed = self._queues[group_id].pop(min_idx)
            logger.debug(
                f"Signal queue overflow for group {group_id}, "
                f"removed lowest weight signal (w={removed.weight:.2f})"
            )

        self._queues[group_id].append(signal)
        logger.debug(
            f"Signal enqueued: type={signal.signal_type.value}, "
            f"group={group_id}, weight={signal.weight:.2f}"
        )
        return True

    def get_signals(self, group_id: str) -> List[Signal]:
        if group_id not in self._queues:
            return []

        now = datetime.now()
        valid = []
        expired_count = 0

        for signal in self._queues[group_id]:
            if signal.expires_at and now >= signal.expires_at:
                expired_count += 1
            else:
                valid.append(signal)

        if expired_count > 0:
            self._queues[group_id] = valid
            logger.debug(
                f"Removed {expired_count} expired signals for group {group_id}"
            )

        return valid

    def clear_session(self, session_key: str) -> int:
        removed = 0
        for group_id in list(self._queues.keys()):
            before = len(self._queues[group_id])
            self._queues[group_id] = [
                s for s in self._queues[group_id]
                if s.session_key != session_key
            ]
            removed += before - len(self._queues[group_id])
            if not self._queues[group_id]:
                del self._queues[group_id]

        if removed > 0:
            logger.debug(f"Cleared {removed} signals for session {session_key}")
        return removed

    def clear_group(self, group_id: str) -> int:
        if group_id not in self._queues:
            return 0
        count = len(self._queues[group_id])
        del self._queues[group_id]
        if count > 0:
            logger.debug(f"Cleared {count} signals for group {group_id}")
        return count

    def update_last_message_time(self, group_id: str) -> None:
        self._last_message_time[group_id] = datetime.now()

    def get_last_message_time(self, group_id: str) -> Optional[datetime]:
        return self._last_message_time.get(group_id)

    def get_silence_duration(self, group_id: str) -> float:
        last_time = self._last_message_time.get(group_id)
        if last_time is None:
            return float("inf")
        return (datetime.now() - last_time).total_seconds()

    def get_active_groups(self) -> List[str]:
        return list(self._queues.keys())

    def aggregate_weight(self, group_id: str) -> float:
        signals = self.get_signals(group_id)
        if not signals:
            return 0.0

        weights = sorted([s.weight for s in signals], reverse=True)
        base = weights[0]
        bonus = sum(w * 0.5 for w in weights[1:])
        return min(1.0, base + bonus)

    @property
    def total_signals(self) -> int:
        return sum(len(signals) for signals in self._queues.values())

    @property
    def group_count(self) -> int:
        return len(self._queues)
