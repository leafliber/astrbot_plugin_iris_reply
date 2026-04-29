from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Dict, List, Optional

from astrbot.api import logger

from .expectation_store import ExpectationStore
from .models import (
    FollowUpDecision,
    FollowUpExpectation,
    FollowUpReplyType,
    ProactiveReplyResult,
)

FollowUpReplyCallback = Callable[
    [ProactiveReplyResult, FollowUpExpectation],
    Coroutine[Any, Any, bool],
]

LLMFollowUpCallback = Callable[
    [FollowUpExpectation],
    Coroutine[Any, Any, Optional[FollowUpDecision]],
]

FOLLOWUP_PROMPT_TEMPLATE = """你是一个在群聊中具有独立人格的 AI 助手。你刚刚在群聊中主动发了一条消息，现在需要判断是否应该继续跟进回复。

【你上次的发言】
{bot_reply_summary}

【用户后续发言】
{aggregated_messages}

【群聊上下文】
{group_context}

请判断：
1. 用户是否在回应你的消息？
2. 你是否需要继续跟进？
3. 如果需要，应该以什么方式回复？

输出 JSON 格式：
{{"should_reply": true/false, "reason": "简短原因", "reply_type": "acknowledge/continue_topic/emotion_support/question", "suggested_direction": "回复方向提示"}}

回复类型说明：
- acknowledge：简单回应/确认，适用于用户简短回复你的消息
- continue_topic：延续话题，适用于用户对你的话题感兴趣
- emotion_support：情感支持，适用于用户表达情感或困扰
- question：提问引导，适用于需要进一步了解用户需求

注意：
- 适度跟进，不要过度保守也不要过度热情
- 如果用户发言与你的话题相关，优先选择跟进
- 如果用户表达困惑、疑问或情感，应该跟进
- 只有在用户明确表示不需要或话题完全无关时才跳过
- 如果用户只是在和其他人聊天，不要强行介入
- 跟进回复要简短自然，不要像客服一样正式"""


def build_followup_prompt(expectation: FollowUpExpectation) -> str:
    aggregated = "\n".join(
        f"  {m.get('sender_name', m.get('user_id', '用户'))}: {m.get('content', '')}"
        for m in expectation.aggregated_messages
    )

    context = "\n".join(
        f"  {m.get('sender_id', '用户')}: {m.get('content', '')}"
        for m in expectation.recent_context[-5:]
    )

    return FOLLOWUP_PROMPT_TEMPLATE.format(
        bot_reply_summary=expectation.bot_reply_summary[:200],
        aggregated_messages=aggregated or "（无）",
        group_context=context or "（无）",
    )


def parse_followup_response(content: str) -> Optional[FollowUpDecision]:
    try:
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            return None

        data = json.loads(text[start:end])

        reply_type_str = data.get("reply_type", "acknowledge")
        try:
            reply_type = FollowUpReplyType(reply_type_str)
        except ValueError:
            reply_type = FollowUpReplyType.ACKNOWLEDGE

        return FollowUpDecision(
            should_reply=bool(data.get("should_reply", False)),
            reason=data.get("reason", ""),
            reply_type=reply_type,
            suggested_direction=data.get("suggested_direction", ""),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse followup response: {e}")
        return None


class FollowUpPlanner:
    def __init__(
        self,
        expectation_store: Optional[ExpectationStore] = None,
        config: Optional[Dict[str, Any]] = None,
        on_followup_reply: Optional[FollowUpReplyCallback] = None,
        on_llm_decide: Optional[LLMFollowUpCallback] = None,
    ) -> None:
        self._config = config or {}
        self._store = expectation_store or ExpectationStore()
        self._on_followup_reply = on_followup_reply
        self._on_llm_decide = on_llm_decide

        self._short_window_timers: Dict[str, asyncio.Task] = {}
        self._window_timeout_timers: Dict[str, asyncio.Task] = {}
        self._closed = False

    def set_followup_reply_callback(self, callback: FollowUpReplyCallback) -> None:
        self._on_followup_reply = callback

    def set_llm_decide_callback(self, callback: LLMFollowUpCallback) -> None:
        self._on_llm_decide = callback

    def create_expectation(
        self,
        session_key: str,
        group_id: str,
        trigger_user_id: str,
        trigger_message: str,
        bot_reply_summary: str,
        followup_count: int = 0,
        recent_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[FollowUpExpectation]:
        if not self._config.get("followup_enabled", True):
            return None

        max_followup = self._config.get("followup_max_count", 3)
        if followup_count >= max_followup:
            logger.debug(
                f"FollowUp count {followup_count} >= max "
                f"{max_followup}, not creating expectation"
            )
            return None

        window_seconds = self._config.get("followup_window_seconds", 150)
        now = datetime.now()

        expectation = FollowUpExpectation(
            session_key=session_key,
            group_id=group_id,
            trigger_user_id=trigger_user_id,
            trigger_message=trigger_message,
            bot_reply_summary=bot_reply_summary,
            followup_window_end=now + timedelta(seconds=window_seconds),
            followup_count=followup_count,
            recent_context=recent_context or [],
        )

        self._cancel_short_window_timer(group_id)
        self._cancel_window_timeout_timer(group_id)

        self._store.put(expectation)

        self._start_window_timeout(group_id, window_seconds)

        logger.info(
            f"FollowUp expectation created: group={group_id}, "
            f"user={trigger_user_id}, count={followup_count}, "
            f"window={window_seconds}s"
        )
        return expectation

    def on_user_message(
        self,
        user_id: str,
        group_id: str,
        message: str,
        sender_name: str = "",
    ) -> bool:
        expectation = self._store.get(group_id)
        if expectation is None:
            return False

        if user_id != expectation.trigger_user_id:
            return False

        if expectation.is_window_expired:
            self._store.remove(group_id)
            self._cancel_short_window_timer(group_id)
            return False

        expectation.aggregated_messages.append({
            "user_id": user_id,
            "sender_name": sender_name,
            "content": message,
            "timestamp": datetime.now().isoformat(),
        })

        short_window = self._config.get("followup_short_window_seconds", 10)
        now = datetime.now()
        new_short_end = now + timedelta(seconds=short_window)

        if new_short_end > expectation.followup_window_end:
            new_short_end = expectation.followup_window_end

        expectation.short_window_end = new_short_end

        remaining = (new_short_end - now).total_seconds()
        self._cancel_short_window_timer(group_id)
        if remaining > 0:
            self._start_short_window_timer(group_id, remaining)

        logger.info(
            f"Message aggregated for group {group_id}: "
            f"total={len(expectation.aggregated_messages)}, "
            f"short_window={remaining:.1f}s"
        )
        return True

    def has_active_expectation(self, group_id: str) -> bool:
        return self._store.has_active(group_id)

    def get_expectation(self, group_id: str) -> Optional[FollowUpExpectation]:
        return self._store.get(group_id)

    def restart_short_window_timer(self, group_id: str, delay: float) -> None:
        self._cancel_short_window_timer(group_id)
        if delay > 0:
            self._start_short_window_timer(group_id, delay)

    def clear_expectation(self, group_id: str) -> None:
        self._cancel_short_window_timer(group_id)
        self._cancel_window_timeout_timer(group_id)
        removed = self._store.remove(group_id)
        if removed:
            logger.debug(f"Expectation cleared for group {group_id}")

    def _start_short_window_timer(self, group_id: str, delay: float) -> None:
        if self._closed:
            return

        self._short_window_timers[group_id] = asyncio.create_task(
            self._short_window_expired(group_id, delay),
            name=f"followup-short-{group_id}",
        )

    def _start_window_timeout(self, group_id: str, delay: float) -> None:
        if self._closed:
            return

        self._window_timeout_timers[group_id] = asyncio.create_task(
            self._followup_window_expired(group_id, delay),
            name=f"followup-window-{group_id}",
        )

    async def _short_window_expired(self, group_id: str, delay: float) -> None:
        try:
            await asyncio.sleep(delay)

            if self._closed:
                return

            expectation = self._store.get(group_id)
            if expectation is None:
                return

            if not expectation.has_aggregated_messages:
                return

            logger.info(
                f"Short window timer fired for group {group_id}: "
                f"messages={len(expectation.aggregated_messages)}"
            )

            self._cancel_window_timeout_timer(group_id)

            await self._trigger_llm_decision(expectation)

        except asyncio.CancelledError:
            logger.debug(f"Short window timer cancelled for group {group_id}")
        except Exception as e:
            logger.error(f"Short window expired error for group {group_id}: {e}")
        finally:
            self._short_window_timers.pop(group_id, None)

    async def _followup_window_expired(self, group_id: str, delay: float) -> None:
        try:
            await asyncio.sleep(delay)

            if self._closed:
                return

            self._window_timeout_timers.pop(group_id, None)

            expectation = self._store.remove(group_id)
            if expectation is None:
                return

            self._cancel_short_window_timer(group_id)

            followup_after_all = self._config.get("followup_after_all_replies", False)
            if expectation.has_aggregated_messages or followup_after_all:
                logger.info(
                    f"FollowUp window expired for group {group_id}: "
                    f"messages={len(expectation.aggregated_messages)}, "
                    f"triggering LLM decision"
                )
                await self._trigger_llm_decision(expectation)
            else:
                logger.debug(
                    f"FollowUp window expired for group {group_id}, "
                    f"no user messages, clearing"
                )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"FollowUp window expired error for group {group_id}: {e}")

    async def _trigger_llm_decision(
        self, expectation: FollowUpExpectation
    ) -> None:
        if expectation._processing:
            logger.debug(
                f"Expectation {expectation.expectation_id} already processing, skipping"
            )
            return

        expectation._processing = True

        try:
            decision: Optional[FollowUpDecision] = None

            if self._on_llm_decide:
                try:
                    decision = await self._on_llm_decide(expectation)
                except Exception as e:
                    logger.warning(f"LLM followup decide failed: {e}")

            if decision is None:
                if self._config.get("followup_fallback_to_rule", True):
                    has_substance = any(
                        len(m.get("content", "")) > 3
                        for m in expectation.aggregated_messages
                    )
                    decision = FollowUpDecision(
                        should_reply=has_substance,
                        reason="规则降级判断",
                        reply_type=FollowUpReplyType.ACKNOWLEDGE,
                    )
                else:
                    decision = FollowUpDecision(should_reply=False)

            if decision.should_reply:
                reply_result = self._build_followup_reply(expectation, decision)

                reply_sent = False
                if self._on_followup_reply:
                    try:
                        reply_sent = await self._on_followup_reply(
                            reply_result, expectation
                        )
                    except Exception as e:
                        logger.error(f"FollowUp reply callback error: {e}")

                if reply_sent:
                    self.create_expectation(
                        session_key=expectation.session_key,
                        group_id=expectation.group_id,
                        trigger_user_id=expectation.trigger_user_id,
                        trigger_message=expectation.trigger_message,
                        bot_reply_summary=reply_result.trigger_prompt[:200],
                        followup_count=expectation.followup_count + 1,
                        recent_context=expectation.recent_context,
                    )
                else:
                    logger.debug(
                        f"FollowUp reply not sent for group {expectation.group_id}, "
                        f"not creating new expectation"
                    )
            else:
                self._store.remove(expectation.group_id)
                logger.debug(
                    f"FollowUp decision: no reply needed for group {expectation.group_id}"
                )

        finally:
            expectation._processing = False

    @staticmethod
    def _build_followup_reply(
        expectation: FollowUpExpectation,
        decision: FollowUpDecision,
    ) -> ProactiveReplyResult:
        aggregated = "\n".join(
            f"  {m.get('sender_name', m.get('user_id', '用户'))}: {m.get('content', '')}"
            for m in expectation.aggregated_messages
        )

        trigger_prompt = (
            "你是一个在群聊中具有独立人格的 AI 助手。你之前主动发起了对话，"
            "现在需要继续跟进回复。\n\n"
            f"【跟进类型】{decision.reply_type.value}\n"
            f"【建议方向】{decision.suggested_direction}\n"
            f"【判断原因】{decision.reason}\n"
        )

        if aggregated:
            trigger_prompt += f"\n【用户后续发言】\n{aggregated}\n"

        trigger_prompt += (
            f"\n【对话对象】{expectation.trigger_user_id}\n"
            "\n【回复要求】\n"
            "- 自然地继续对话，绝对不要提及'跟进'、'系统检测'、'信号'等元信息\n"
            "- 根据用户的反应调整回复策略\n"
            "- 如果用户似乎不感兴趣，简短回应即可，不要强行延续话题\n"
            "- 语气要符合你的人格设定，保持自然随意\n"
            "- 回复要简短（1-2句话），不要像客服一样正式\n"
            "- 如果觉得不应该继续跟进，回复'PASS'表示跳过\n"
        )

        return ProactiveReplyResult(
            trigger_prompt=trigger_prompt,
            reply_params={"max_tokens": 150, "temperature": 0.7},
            reason=f"FollowUp: {decision.reason}",
            group_id=expectation.group_id,
            session_key=expectation.session_key,
            target_user=expectation.trigger_user_id,
            recent_messages=expectation.recent_context,
            source="followup",
        )

    def _cancel_short_window_timer(self, group_id: str) -> None:
        timer = self._short_window_timers.pop(group_id, None)
        if timer and not timer.done():
            timer.cancel()

    def _cancel_window_timeout_timer(self, group_id: str) -> None:
        timer = self._window_timeout_timers.pop(group_id, None)
        if timer and not timer.done():
            timer.cancel()

    @property
    def active_expectation_count(self) -> int:
        return self._store.active_count

    async def close(self) -> None:
        self._closed = True

        for timer in list(self._short_window_timers.values()):
            timer.cancel()
        for timer in list(self._window_timeout_timers.values()):
            timer.cancel()

        for timer in list(self._short_window_timers.values()):
            try:
                await timer
            except asyncio.CancelledError:
                pass
        for timer in list(self._window_timeout_timers.values()):
            try:
                await timer
            except asyncio.CancelledError:
                pass

        self._short_window_timers.clear()
        self._window_timeout_timers.clear()
        self._store.clear()
        logger.debug("FollowUpPlanner closed")
