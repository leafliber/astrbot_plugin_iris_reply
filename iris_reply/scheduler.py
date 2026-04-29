from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from astrbot.api import logger

from .models import AggregatedDecision, Signal
from .signal_engine import SignalQueue

ReplyCallback = Callable[[AggregatedDecision], Coroutine[Any, Any, None]]

LLMConfirmCallback = Callable[
    [str, List[Signal], List[Dict[str, Any]]],
    Coroutine[Any, Any, bool],
]


class GroupScheduler:
    def __init__(
        self,
        signal_queue: SignalQueue,
        config: Dict[str, Any],
        on_reply: Optional[ReplyCallback] = None,
        on_llm_confirm: Optional[LLMConfirmCallback] = None,
    ) -> None:
        self._config = config
        self._signal_queue = signal_queue
        self._on_reply = on_reply
        self._on_llm_confirm = on_llm_confirm

        self._timers: Dict[str, asyncio.Task] = {}
        self._active_groups: Set[str] = set()
        self._closed = False

    def set_reply_callback(self, callback: ReplyCallback) -> None:
        self._on_reply = callback

    def set_llm_confirm_callback(self, callback: LLMConfirmCallback) -> None:
        self._on_llm_confirm = callback

    def ensure_timer(self, group_id: str) -> None:
        if self._closed:
            return

        if group_id in self._timers:
            task = self._timers[group_id]
            if not task.done():
                return

        self._timers[group_id] = asyncio.create_task(
            self._timer_loop(group_id),
            name=f"proactive-timer-{group_id}",
        )
        self._active_groups.add(group_id)
        logger.debug(f"Timer created for group {group_id}")

    async def _timer_loop(self, group_id: str) -> None:
        check_interval = self._config.get("signal_check_interval_seconds", 30)
        silence_timeout = self._config.get("signal_silence_timeout_seconds", 600)

        try:
            while not self._closed:
                await asyncio.sleep(check_interval)

                if self._closed:
                    break

                silence = self._signal_queue.get_silence_duration(group_id)
                if silence >= silence_timeout:
                    logger.debug(
                        f"Group {group_id} silence timeout "
                        f"({silence:.0f}s >= {silence_timeout}s), "
                        f"destroying timer"
                    )
                    self._signal_queue.clear_group(group_id)
                    break

                min_silence = self._config.get("signal_min_silence_seconds", 60)
                if silence < min_silence:
                    continue

                signals = self._signal_queue.get_signals(group_id)
                if not signals:
                    continue

                await self._aggregate_and_decide(group_id, signals)

        except asyncio.CancelledError:
            logger.debug(f"Timer cancelled for group {group_id}")
        except Exception as e:
            logger.error(f"Timer error for group {group_id}: {e}")
        finally:
            self._active_groups.discard(group_id)
            self._timers.pop(group_id, None)

    async def _aggregate_and_decide(
        self, group_id: str, signals: List[Signal]
    ) -> None:
        aggregated_weight = self._signal_queue.aggregate_weight(group_id)

        best_signal = max(signals, key=lambda s: s.weight)
        target_user_id = best_signal.user_id
        session_key = best_signal.session_key

        recent_messages: List[Dict[str, Any]] = []
        for s in signals:
            preview = s.metadata.get("text_preview", "")
            if preview:
                recent_messages.append({
                    "sender_id": s.user_id,
                    "content": preview,
                    "timestamp": s.created_at.isoformat(),
                })

        weight_direct = self._config.get("signal_weight_direct_reply", 0.8)
        weight_llm = self._config.get("signal_weight_llm_confirm", 0.5)
        proactive_mode = self._config.get("proactive_mode", "rule")

        if aggregated_weight >= weight_direct:
            decision = AggregatedDecision(
                should_reply=True,
                session_key=session_key,
                group_id=group_id,
                target_user_id=target_user_id,
                aggregated_weight=aggregated_weight,
                signals=list(signals),
                reason=f"聚合权重 {aggregated_weight:.2f} >= {weight_direct}",
                recent_messages=recent_messages,
                llm_confirmed=False,
            )
            await self._execute_reply(decision)

        elif (
            aggregated_weight >= weight_llm
            and proactive_mode == "hybrid"
        ):
            should_reply = await self._try_llm_confirm(
                group_id, signals, recent_messages
            )
            if should_reply:
                decision = AggregatedDecision(
                    should_reply=True,
                    session_key=session_key,
                    group_id=group_id,
                    target_user_id=target_user_id,
                    aggregated_weight=aggregated_weight,
                    signals=list(signals),
                    reason=f"聚合权重 {aggregated_weight:.2f} >= {weight_llm} (LLM 确认)",
                    recent_messages=recent_messages,
                    llm_confirmed=True,
                )
                await self._execute_reply(decision)
            else:
                self._signal_queue.clear_group(group_id)

        else:
            self._signal_queue.clear_group(group_id)

    async def _execute_reply(self, decision: AggregatedDecision) -> None:
        if self._on_reply:
            try:
                await self._on_reply(decision)
            except Exception as e:
                logger.error(f"Reply callback error: {e}")

    async def _try_llm_confirm(
        self,
        group_id: str,
        signals: List[Signal],
        recent_messages: List[Dict[str, Any]],
    ) -> bool:
        if not self._on_llm_confirm:
            return False

        try:
            return await self._on_llm_confirm(group_id, signals, recent_messages)
        except Exception as e:
            logger.warning(f"LLM confirm callback failed: {e}")
            return False

    @property
    def active_group_count(self) -> int:
        return len(self._active_groups)

    async def close(self) -> None:
        self._closed = True

        for group_id, task in list(self._timers.items()):
            task.cancel()

        for group_id, task in list(self._timers.items()):
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._timers.clear()
        self._active_groups.clear()
        logger.debug("GroupScheduler closed")
