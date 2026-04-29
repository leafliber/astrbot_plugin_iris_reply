from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from astrbot.api import logger

from .cooldown import CooldownManager
from .followup import FollowUpPlanner
from .models import (
    AggregatedDecision,
    ProactiveReplyResult,
    ReplyDecision,
    ReplyDecisionType,
)
from .scheduler import GroupScheduler
from .signal_engine import SignalGenerator, SignalQueue


PROACTIVE_PROMPT_TEMPLATE = """你是一个在群聊中具有独立人格的 AI 助手。你正在主动加入一段群聊对话，请根据上下文生成一条自然的回复消息。

【触发原因】
{reason}

【群聊近期消息】
{recent_messages}

【目标对话对象】
{target_user}

【信号详情】
{signal_details}

【回复要求】
- 你是在群聊中主动发言，不是被 @ 或被直接提问，所以语气要自然随意
- 绝对不要提及"系统检测"、"信号"、"触发"等任何元信息
- 回复要简短（1-3句话），像真人一样参与群聊
- 根据触发原因选择合适的回复方式：
  - 问题信号：可以尝试回答、提供思路或引导讨论
  - 情感信号：给予关心、共鸣或安慰，语气温暖但不过度
  - 注意力信号：主动打招呼、找话题或回应氛围
  - 提及信号：回应对方的请求或表达
- 保持你的人格设定和说话风格
- 不要过度热情，保持适度距离感
- 如果不确定该说什么，回复"PASS"表示跳过本次回复"""


def build_proactive_prompt(decision: AggregatedDecision) -> str:
    recent = "\n".join(
        f"  {m.get('sender_id', '用户')}: {m.get('content', '')}"
        for m in decision.recent_messages[-10:]
    )

    signal_details = "\n".join(
        f"  - 类型: {s.signal_type.value}, 权重: {s.weight:.2f}"
        for s in decision.signals
    )

    return PROACTIVE_PROMPT_TEMPLATE.format(
        reason=decision.reason,
        recent_messages=recent or "（无近期消息）",
        target_user=decision.target_user_id or "全体",
        signal_details=signal_details or "（无）",
    )


class ReplyCoordinator:
    def __init__(
        self,
        signal_generator: SignalGenerator,
        signal_queue: SignalQueue,
        scheduler: GroupScheduler,
        cooldown_manager: CooldownManager,
        followup_planner: FollowUpPlanner,
        config: Dict[str, Any],
    ) -> None:
        self._signal_generator = signal_generator
        self._signal_queue = signal_queue
        self._scheduler = scheduler
        self._cooldown = cooldown_manager
        self._followup = followup_planner
        self._config = config

    def on_message(
        self,
        text: str,
        user_id: str,
        group_id: str,
        session_key: str,
        is_bot: bool = False,
        emotion_intensity: float = 0.0,
    ) -> ReplyDecision:
        if not self._config.get("proactive_enabled", True):
            return ReplyDecision(
                decision_type=ReplyDecisionType.SKIP,
                reason="主动回复功能已禁用",
            )

        if self._cooldown.is_active(group_id):
            return ReplyDecision(
                decision_type=ReplyDecisionType.SKIP,
                reason="群组处于冷却中",
            )

        if self._is_quiet_hours():
            return ReplyDecision(
                decision_type=ReplyDecisionType.SKIP,
                reason="当前处于静音时段",
            )

        if is_bot:
            return ReplyDecision(
                decision_type=ReplyDecisionType.SKIP,
                reason="忽略自身消息",
            )

        if self._followup.has_active_expectation(group_id):
            handled = self._followup.on_user_message(
                user_id=user_id,
                group_id=group_id,
                message=text,
            )
            if handled:
                return ReplyDecision(
                    decision_type=ReplyDecisionType.FOLLOWUP_AGGREGATED,
                    reason="消息已聚合到跟进窗口",
                )

        self._signal_queue.update_last_message_time(group_id)

        signals = self._signal_generator.generate(
            text=text,
            user_id=user_id,
            group_id=group_id,
            session_key=session_key,
            emotion_intensity=emotion_intensity,
        )

        if not signals:
            return ReplyDecision(
                decision_type=ReplyDecisionType.NO_SIGNAL,
                reason="未检测到有效信号",
            )

        for signal in signals:
            self._signal_queue.enqueue(signal)

        self._scheduler.ensure_timer(group_id)

        return ReplyDecision(
            decision_type=ReplyDecisionType.SIGNAL_ENQUEUED,
            reason=f"检测到 {len(signals)} 个信号，已入队等待聚合",
        )

    async def handle_aggregated_decision(
        self, decision: AggregatedDecision
    ) -> Optional[ProactiveReplyResult]:
        if self._cooldown.is_active(decision.group_id):
            logger.debug(
                f"Group {decision.group_id} in cooldown, skipping proactive reply"
            )
            return None

        if self._is_quiet_hours():
            logger.debug("Quiet hours active, skipping proactive reply")
            return None

        trigger_prompt = build_proactive_prompt(decision)

        max_tokens = self._config.get("proactive_max_tokens", 200)
        temperature = self._config.get("proactive_temperature", 0.8)

        result = ProactiveReplyResult(
            trigger_prompt=trigger_prompt,
            reply_params={"max_tokens": max_tokens, "temperature": temperature},
            reason=decision.reason,
            group_id=decision.group_id,
            session_key=decision.session_key,
            target_user=decision.target_user_id,
            recent_messages=decision.recent_messages,
            source="proactive",
        )

        logger.info(
            f"Proactive reply triggered: group={decision.group_id}, "
            f"weight={decision.aggregated_weight:.2f}, "
            f"reason={decision.reason}"
        )

        return result

    async def handle_followup_reply(
        self,
        result: ProactiveReplyResult,
        expectation: Any,
    ) -> bool:
        if self._cooldown.is_active(result.group_id):
            logger.debug(
                f"Group {result.group_id} in cooldown, skipping followup reply"
            )
            return False

        if self._is_quiet_hours():
            logger.debug("Quiet hours active, skipping followup reply")
            return False

        logger.info(
            f"FollowUp reply triggered: group={result.group_id}, "
            f"user={result.target_user}, reason={result.reason}"
        )
        return True

    def on_reply_sent(
        self,
        group_id: str,
        session_key: str,
        target_user_id: str,
        bot_reply_summary: str,
        trigger_message: str,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        source: str = "proactive",
    ) -> None:
        self._signal_queue.clear_group(group_id)

        if source == "proactive":
            self._followup.create_expectation(
                session_key=session_key,
                group_id=group_id,
                trigger_user_id=target_user_id,
                trigger_message=trigger_message,
                bot_reply_summary=bot_reply_summary,
                recent_context=recent_context,
            )

    def _is_quiet_hours(self) -> bool:
        quiet_enabled = self._config.get("quiet_hours_enabled", False)
        if not quiet_enabled:
            return False

        quiet_start = self._config.get("quiet_hours_start", "23:00")
        quiet_end = self._config.get("quiet_hours_end", "07:00")

        try:
            now = datetime.now()
            current_minutes = now.hour * 60 + now.minute

            start_parts = quiet_start.split(":")
            end_parts = quiet_end.split(":")
            start_minutes = int(start_parts[0]) * 60 + int(start_parts[1])
            end_minutes = int(end_parts[0]) * 60 + int(end_parts[1])

            if start_minutes <= end_minutes:
                return start_minutes <= current_minutes < end_minutes
            else:
                return current_minutes >= start_minutes or current_minutes < end_minutes
        except (ValueError, IndexError):
            return False

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "signal_queue_total": self._signal_queue.total_signals,
            "signal_queue_groups": self._signal_queue.group_count,
            "scheduler_active_groups": self._scheduler.active_group_count,
            "cooldown_active_groups": self._cooldown.active_count,
            "followup_active_expectations": self._followup.active_expectation_count,
        }
