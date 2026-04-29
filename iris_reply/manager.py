from __future__ import annotations

from typing import Any, Dict, List, Optional

from astrbot.api import logger
from astrbot.api.star import Context

from .cooldown import CooldownManager
from .cooperation import MemoryCooperation
from .coordinator import ReplyCoordinator
from .expectation_store import ExpectationStore
from .followup import FollowUpPlanner
from .models import (
    AggregatedDecision,
    ProactiveReplyResult,
    ReplyDecision,
    ReplyDecisionType,
)
from .scheduler import GroupScheduler
from .sender import ReplySender
from .signal_engine import SignalGenerator, SignalQueue


class ReplyManager:
    def __init__(self, context: Context, config: Dict[str, Any]) -> None:
        self._context = context
        self._config = config

        self._signal_generator = SignalGenerator(config)
        self._signal_queue = SignalQueue(config)
        self._cooldown = CooldownManager(
            default_duration=config.get("cooldown_default_minutes", 20),
            max_groups=1000,
        )
        self._expectation_store = ExpectationStore()
        self._followup = FollowUpPlanner(
            expectation_store=self._expectation_store,
            config=config,
        )
        self._scheduler = GroupScheduler(
            signal_queue=self._signal_queue,
            config=config,
        )
        self._cooperation = MemoryCooperation(context, config)
        self._sender = ReplySender(context)
        self._coordinator = ReplyCoordinator(
            signal_generator=self._signal_generator,
            signal_queue=self._signal_queue,
            scheduler=self._scheduler,
            cooldown_manager=self._cooldown,
            followup_planner=self._followup,
            config=config,
        )

        self._scheduler.set_reply_callback(self._on_aggregated_decision)
        self._scheduler.set_llm_confirm_callback(self._on_llm_confirm)

        self._followup.set_followup_reply_callback(self._on_followup_reply)
        self._followup.set_llm_decide_callback(self._on_llm_followup_decide)

        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return

        self._initialized = True
        logger.info("ReplyManager initialized")

    def store_umo(self, group_id: str, unified_msg_origin: str) -> None:
        self._sender.store_umo(group_id, unified_msg_origin)

    async def on_message(
        self,
        text: str,
        user_id: str,
        group_id: str,
        session_key: str,
        is_bot: bool = False,
        sender_name: str = "",
    ) -> ReplyDecision:
        decision = self._coordinator.on_message(
            text=text,
            user_id=user_id,
            group_id=group_id,
            session_key=session_key,
            is_bot=is_bot,
            emotion_intensity=0.0,
        )

        if decision.decision_type == ReplyDecisionType.SIGNAL_ENQUEUED:
            logger.info(
                f"Message processed: group={group_id}, "
                f"user={user_id}, decision={decision.decision_type.value}, "
                f"reason={decision.reason}"
            )

        return decision

    async def _on_aggregated_decision(
        self, decision: AggregatedDecision
    ) -> None:
        result = await self._coordinator.handle_aggregated_decision(decision)
        if result is None:
            return

        sent = await self._sender.send_proactive_reply(
            result, llm_call=self._llm_call
        )

        if sent:
            self._coordinator.on_reply_sent(
                group_id=result.group_id,
                session_key=result.session_key,
                target_user_id=result.target_user or "",
                bot_reply_summary=result.trigger_prompt[:200],
                trigger_message="",
                recent_context=result.recent_messages,
                source="proactive",
            )

    async def _on_llm_confirm(
        self,
        group_id: str,
        signals: List[Any],
        recent_messages: List[Dict[str, Any]],
    ) -> bool:
        return await self._cooperation.llm_confirm_proactive(
            group_id, signals, recent_messages
        )

    async def _on_followup_reply(
        self,
        result: ProactiveReplyResult,
        expectation: Any,
    ) -> bool:
        approved = await self._coordinator.handle_followup_reply(
            result, expectation
        )
        if not approved:
            return False

        sent = await self._sender.send_proactive_reply(
            result, llm_call=self._llm_call
        )

        if sent:
            self._coordinator.on_reply_sent(
                group_id=result.group_id,
                session_key=result.session_key,
                target_user_id=result.target_user or "",
                bot_reply_summary=result.trigger_prompt[:200],
                trigger_message="",
                recent_context=result.recent_messages,
                source="followup",
            )

        return sent

    async def _on_llm_followup_decide(
        self, expectation: Any
    ) -> Optional[Any]:
        return await self._cooperation.llm_decide_followup(expectation)

    async def _llm_call(self, prompt: str, **kwargs: Any) -> Optional[str]:
        try:
            provider = None
            provider_id = self._config.get("provider_id", "")
            if provider_id:
                try:
                    provider = self._context.get_using_provider(provider_id)
                except Exception:
                    provider = None

            if provider is None:
                provider = self._context.get_using_provider()

            if provider is None:
                return None

            response = await provider.text_chat(
                prompt=prompt,
                session_id="proactive_reply_system",
            )

            if hasattr(response, "completion_text"):
                return response.completion_text
            if isinstance(response, str):
                return response
            return str(response) if response else None

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def activate_cooldown(
        self,
        group_id: str,
        duration_minutes: Optional[int] = None,
        reason: Optional[str] = None,
        initiated_by: str = "user",
    ) -> str:
        return self._cooldown.activate(
            group_id=group_id,
            duration_minutes=duration_minutes,
            reason=reason,
            initiated_by=initiated_by,
        )

    def deactivate_cooldown(self, group_id: str) -> str:
        return self._cooldown.deactivate(group_id)

    def get_cooldown_status(self, group_id: str) -> Optional[Dict[str, Any]]:
        state = self._cooldown.get_state(group_id)
        if state is None:
            return None

        return {
            "group_id": state.group_id,
            "is_active": self._cooldown.is_active(group_id),
            "started_at": state.started_at.isoformat(),
            "expires_at": state.expires_at.isoformat(),
            "initiated_by": state.initiated_by,
            "reason": state.reason,
        }

    def update_config(self, key: str, value: Any) -> None:
        self._config[key] = value

    def get_config(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    @property
    def status(self) -> Dict[str, Any]:
        coordinator_status = self._coordinator.status
        return {
            **coordinator_status,
            "proactive_enabled": self._config.get("proactive_enabled", True),
            "proactive_mode": self._config.get("proactive_mode", "rule"),
            "followup_enabled": self._config.get("followup_enabled", True),
            "quiet_hours_enabled": self._config.get("quiet_hours_enabled", False),
        }

    async def close(self) -> None:
        await self._scheduler.close()
        await self._followup.close()
        logger.info("ReplyManager closed")
