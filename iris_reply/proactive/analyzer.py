from __future__ import annotations

import logging
from typing import Any

from iris_reply.core.llm_client import LLMClient
from iris_reply.core.memory_api import MemoryAPI
from iris_reply.models.models import AnalysisResult
from iris_reply.config.config import ReplyConfig

logger = logging.getLogger("iris_reply.analyzer")

_HELP_PATTERNS = {"帮我", "帮忙", "求助", "怎么办", "请问", "有没有人", "谁能", "怎么搞", "不懂", "不会", "教教", "求教"}
_QUESTION_PATTERNS = {"吗", "呢", "？", "?", "怎么", "如何", "为什么", "啥", "什么"}
_EMOTION_PATTERNS = {"郁闷", "烦", "难过", "崩溃", "焦虑", "无语", "气死", "受不了", "好累", "心累"}


class MessageAnalyzer:
    def __init__(
        self,
        llm_client: LLMClient,
        memory_api: MemoryAPI,
        config: ReplyConfig,
    ):
        self._llm = llm_client
        self._memory = memory_api
        self._config = config

    async def analyze(self, messages: list[dict[str, Any]], group_id: str) -> AnalysisResult:
        mode = self._config.proactive.get("mode", "rule")

        if mode == "rule":
            return self._rule_analyze(messages)
        elif mode == "llm":
            return await self._llm_analyze(messages, group_id)
        else:
            rule_result = self._rule_analyze(messages)
            if rule_result.should_consider:
                return await self._llm_analyze(messages, group_id)
            return rule_result

    def _rule_analyze(self, messages: list[dict[str, Any]]) -> AnalysisResult:
        if not messages:
            return AnalysisResult()

        result = AnalysisResult()
        window = self._config.proactive.get("analysis_window_messages", 10)
        recent = messages[-window:]

        topic_relevance = 0.0
        intent_signal = 0.0
        emotion_signal = 0.0
        mention_signal = 0.0

        for msg in recent:
            content = str(msg.get("content", ""))
            if not content:
                continue

            for pattern in _HELP_PATTERNS:
                if pattern in content:
                    intent_signal = max(intent_signal, 0.7)
                    break

            for pattern in _QUESTION_PATTERNS:
                if pattern in content:
                    intent_signal = max(intent_signal, 0.4)
                    break

            for pattern in _EMOTION_PATTERNS:
                if pattern in content:
                    emotion_signal = max(emotion_signal, 0.6)
                    break

        silence_gap = self._calc_silence_gap(recent)

        relevance_threshold = self._config.proactive.get("relevance_threshold", 0.6)
        intent_threshold = self._config.proactive.get("intent_threshold", 0.5)
        silence_threshold = self._config.proactive.get("silence_gap_seconds", 120)

        should_consider = (
            topic_relevance >= relevance_threshold
            or intent_signal >= intent_threshold
            or emotion_signal >= 0.5
            or silence_gap >= silence_threshold
        )

        result.should_consider = should_consider
        result.topic_relevance = topic_relevance
        result.intent_signal = intent_signal
        result.emotion_signal = emotion_signal
        result.silence_gap = silence_gap
        result.mention_signal = mention_signal
        result.summary = self._summarize(recent)

        return result

    async def _llm_analyze(self, messages: list[dict[str, Any]], group_id: str) -> AnalysisResult:
        rule_result = self._rule_analyze(messages)

        recent_text = "\n".join(
            f"[{m.get('sender_name', m.get('role', '?'))}]: {m.get('content', '')}"
            for m in messages[-10:]
        )

        system_prompt = (
            "你是一个群聊消息分析助手。分析以下消息流，判断 Bot 是否应该主动参与对话。\n"
            "你必须只输出 JSON，不要输出任何其他内容。"
        )
        prompt = f"""## 近期消息
{recent_text or '（无）'}

## 规则预分析结果
- 话题相关性: {rule_result.topic_relevance:.2f}
- 意图信号: {rule_result.intent_signal:.2f}
- 情感信号: {rule_result.emotion_signal:.2f}
- 沉默间隔: {rule_result.silence_gap:.0f}秒

请分析消息流，输出 JSON：
{{
    "should_consider": true或false,
    "topic_relevance": 0.0到1.0,
    "intent_signal": 0.0到1.0,
    "emotion_signal": 0.0到1.0,
    "mention_signal": 0.0到1.0,
    "summary": "消息流摘要"
}}"""

        provider_id = self._config.proactive.get("llm_provider_id", "")
        result = await self._llm.generate_json(
            prompt, system_prompt, provider_id, module="proactive_analyze"
        )

        if result is None:
            return rule_result

        return AnalysisResult(
            should_consider=result.get("should_consider", rule_result.should_consider),
            topic_relevance=float(result.get("topic_relevance", rule_result.topic_relevance)),
            intent_signal=float(result.get("intent_signal", rule_result.intent_signal)),
            emotion_signal=float(result.get("emotion_signal", rule_result.emotion_signal)),
            silence_gap=rule_result.silence_gap,
            mention_signal=float(result.get("mention_signal", rule_result.mention_signal)),
            summary=result.get("summary", rule_result.summary),
        )

    def _calc_silence_gap(self, messages: list[dict[str, Any]]) -> float:
        if not messages:
            return 0.0

        timestamps: list[float] = []
        for msg in messages:
            ts = msg.get("timestamp")
            if ts is not None and isinstance(ts, (int, float)):
                timestamps.append(float(ts))

        if len(timestamps) < 2:
            return 0.0

        timestamps.sort()

        max_gap = 0.0
        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i - 1]
            max_gap = max(max_gap, gap)

        return max_gap

    def _summarize(self, messages: list[dict[str, Any]]) -> str:
        if not messages:
            return ""
        parts = []
        for msg in messages[-5:]:
            sender = msg.get("sender_name", msg.get("role", "?"))
            content = str(msg.get("content", ""))[:50]
            if content:
                parts.append(f"{sender}: {content}")
        return "\n".join(parts)
