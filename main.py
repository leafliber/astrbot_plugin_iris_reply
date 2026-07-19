from __future__ import annotations

import asyncio
import time
from typing import Any

from astrbot.api import logger
from astrbot.api.event.filter import (
    EventMessageType,
    after_message_sent,
    command_group,
    event_message_type,
    llm_tool,
    on_llm_request,
    on_llm_response,
    permission_type,
)
from astrbot.api.event.filter import PermissionType
from astrbot.api.star import Context, Star
from astrbot.core.agent.message import TextPart
from astrbot.core.provider.entities import LLMResponse, ProviderRequest

from .iris_reply.admin import AdminCommands
from .iris_reply.api import register_web_apis, sync_stats_group_state
from .iris_reply.config import ConfigManager
from .iris_reply.decision import DecisionCore, DecisionRequest
from .iris_reply.perception import ContextPackager, Gatekeeper, SlidingWindow, WindowMessage
from .iris_reply.prompts import SPEAK_HINTS
from .iris_reply.proactive import ProactiveEngine
from .iris_reply.signals import SignalGate
from .iris_reply.state import StateManager
from .iris_reply.stats import StatsCollector
from .iris_reply.tools import ToolContext

PLUGIN_NAME = "astrbot_plugin_iris_reply"

_IRIS_ACTIVE_TIMEOUT = 120
_UMO_KV_KEY = "iris_reply:group_umo"


class IrisReply(Star):

    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context, config)
        self._config = ConfigManager(config if config else context.get_config())
        self._state = StateManager(self._config)
        self._gatekeeper = Gatekeeper(self._config, self._state)
        self._sliding_window = SlidingWindow(self._config)
        self._context_packager = ContextPackager(self._config)
        self._signals = SignalGate(self._config, self._state)
        self._decision_core = DecisionCore(
            self._config, self._state, self._sliding_window, self._context_packager,
        )
        self._tool_ctx = ToolContext()
        self._admin = AdminCommands(self._state)
        self._stats = StatsCollector()
        self._reply_in_progress: dict[str, float] = {}
        self._passive_active: dict[str, float] = {}
        self._triggering: dict[str, float] = {}
        self._follow_pending: set[str] = set()
        self._group_umo: dict[str, str] = {}
        self._umo_dirty: bool = False
        self._self_id: str = ""
        self._save_task: asyncio.Task | None = None
        self._save_interval = 30
        self._proactive = ProactiveEngine(
            self.context,
            self._config,
            self._state,
            self._sliding_window,
            self._signals,
            self._decision_core,
            self._stats,
            umo_get=lambda gid: self._group_umo.get(gid),
            is_busy=self._is_busy,
            self_id_get=lambda: self._self_id,
            save_fn=lambda: self._state.save_dirty(self._kv_save),
        )

    async def initialize(self) -> None:
        await self._state.load_all(self._kv_load)
        umo_data = await self._kv_load(_UMO_KV_KEY)
        if isinstance(umo_data, dict):
            self._group_umo = {str(k): str(v) for k, v in umo_data.items()}
        config_overrides = await self._kv_load("iris_reply:config_overrides")
        self._config.load_overrides(config_overrides)
        self._save_task = asyncio.create_task(self._periodic_save())
        self._stats.enabled = self._config.stats_enabled
        register_web_apis(
            context=self.context,
            plugin_name=PLUGIN_NAME,
            config=self._config,
            state=self._state,
            stats=self._stats,
            window=self._sliding_window,
            kv_save=self._kv_save,
        )
        await self._proactive.start()
        logger.info("Iris Reply: initialized")

    async def terminate(self) -> None:
        await self._proactive.stop()
        if self._save_task and not self._save_task.done():
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
        await self._state.save_all(self._kv_save)
        await self._kv_save("iris_reply:config_overrides", self._config.get_overrides())
        await self._kv_save(_UMO_KV_KEY, dict(self._group_umo))
        self._follow_pending.clear()
        self._reply_in_progress.clear()
        self._passive_active.clear()
        self._triggering.clear()
        logger.info("Iris Reply: terminated")

    async def _periodic_save(self) -> None:
        while True:
            await asyncio.sleep(self._save_interval)
            try:
                await self._state.save_dirty(self._kv_save)
                await self._kv_save("iris_reply:config_overrides", self._config.get_overrides())
                if self._umo_dirty:
                    self._umo_dirty = False
                    await self._kv_save(_UMO_KV_KEY, dict(self._group_umo))
                self._sliding_window.cleanup(self._state.get_whitelist())
                self._cleanup_stale_active()
                sync_stats_group_state(self._state, self._stats)
            except Exception as e:
                logger.warning("Iris Reply: periodic save error: %s", e)

    def _cleanup_stale_active(self) -> None:
        now = time.time()
        stale_rip = [gid for gid, ts in self._reply_in_progress.items() if now - ts > _IRIS_ACTIVE_TIMEOUT]
        for gid in stale_rip:
            logger.info("Iris Reply: cleaning up stale reply_in_progress for group %s (timeout)", gid)
            self._reply_in_progress.pop(gid, None)
        stale_passive = [gid for gid, ts in self._passive_active.items() if now - ts > _IRIS_ACTIVE_TIMEOUT]
        for gid in stale_passive:
            logger.info("Iris Reply: cleaning up stale passive for group %s (timeout)", gid)
            self._passive_active.pop(gid, None)
        stale_triggering = [gid for gid, ts in self._triggering.items()
                            if gid not in self._reply_in_progress and now - ts > _IRIS_ACTIVE_TIMEOUT]
        for gid in stale_triggering:
            logger.info("Iris Reply: cleaning up stale triggering for group %s", gid)
            self._triggering.pop(gid, None)

    def _is_busy(self, group_id: str) -> bool:
        return (
            group_id in self._reply_in_progress
            or group_id in self._triggering
            or group_id in self._passive_active
        )

    async def _kv_save(self, key: str, value: Any) -> None:
        await self.put_kv_data(key, value)

    async def _kv_load(self, key: str) -> Any:
        return await self.get_kv_data(key, None)

    def _get_group_id(self, event) -> str | None:
        group_id = event.get_group_id()
        if not group_id:
            event.set_result("无法获取群ID")
            return None
        return group_id

    async def _get_provider_id(self, event, preferred: str = "") -> str | None:
        if preferred:
            return preferred
        try:
            return await self.context.get_current_chat_provider_id(
                event.unified_msg_origin
            )
        except Exception:
            logger.error("Iris Reply: failed to get provider ID")
            return None

    @llm_tool(name="add_follow_up")
    async def tool_add_follow_up(self, event, user_ids: str = "") -> str:
        """当你希望持续关注某些用户的发言时调用此工具。将在后续消息中匹配指定用户时自动触发回复。

        Args:
            user_ids(string): 逗号分隔的用户ID列表，如 "user1,user2"
        """
        group_id = self._tool_ctx.current_group_id or event.get_group_id()
        if not group_id:
            return "error: no group context"

        uid_list = [u.strip() for u in user_ids.split(",") if u.strip()] if user_ids else None

        if not uid_list:
            return "error: must provide at least one user_id"

        if len(uid_list) > 10:
            return "error: too many user_ids (max 10 per call)"

        async with self._state.get_lock(group_id):
            self._state.add_anchor_watch(group_id, users=uid_list)
        logger.debug("Iris Reply: add_follow_up for group %s, users=%s", group_id, uid_list)
        return f"ok: following users={uid_list}"

    @llm_tool(name="end_follow_up")
    async def tool_end_follow_up(self, event, user_ids: str = "") -> str:
        """当你不再需要关注某些用户时调用此工具，移除对应的跟进记录。不提供参数则移除所有跟进记录。

        Args:
            user_ids(string): 逗号分隔的用户ID列表，如 "user1,user2"
        """
        group_id = self._tool_ctx.current_group_id or event.get_group_id()
        if not group_id:
            return "error: no group context"

        uid_list = [u.strip() for u in user_ids.split(",") if u.strip()] if user_ids else None

        async with self._state.get_lock(group_id):
            self._state.remove_anchor_watch(group_id, user_ids=uid_list)
        logger.debug("Iris Reply: end_follow_up for group %s, users=%s", group_id, uid_list)
        return f"ok: removed follow-up users={uid_list}"

    @llm_tool(name="set_cooldown")
    async def tool_set_cooldown(self, event, minutes: int = 5) -> str:
        """当你认为应该暂时停止主动回复时调用此工具。设置冷却时间，冷却期间不会主动触发任何回复。

        Args:
            minutes(number): 冷却时间（分钟），范围 1-120，默认 5
        """
        group_id = self._tool_ctx.current_group_id or event.get_group_id()
        if not group_id:
            return "error: no group context"

        async with self._state.get_lock(group_id):
            actual = self._state.set_cooldown(group_id, minutes)
        logger.debug("Iris Reply: set_cooldown for group %s, %d min", group_id, actual)
        return f"ok: cooldown set for {actual} minutes"

    @command_group("iris_reply")
    @permission_type(PermissionType.ADMIN)
    @event_message_type(EventMessageType.GROUP_MESSAGE)
    def iris(self):
        pass

    @iris.command("enable")
    async def cmd_enable(self, event) -> None:
        group_id = self._get_group_id(event)
        if not group_id:
            return
        self._state.add_to_whitelist(group_id)
        await self._state.save_dirty(self._kv_save)
        event.set_result(f"群 {group_id} 已启用 Iris Reply")

    @iris.command("disable")
    async def cmd_disable(self, event) -> None:
        group_id = self._get_group_id(event)
        if not group_id:
            return
        self._state.remove_from_whitelist(group_id)
        self._sliding_window.remove_group(group_id)
        self._state.remove_group_lock(group_id)
        await self._state.save_dirty(self._kv_save)
        event.set_result(f"群 {group_id} 已禁用 Iris Reply")

    @iris.command("status")
    async def cmd_status(self, event) -> None:
        group_id = self._get_group_id(event)
        if not group_id:
            return
        text = self._admin.get_status(group_id)
        event.set_result(text)

    @iris.command("reset")
    async def cmd_reset(self, event) -> None:
        group_id = self._get_group_id(event)
        if not group_id:
            return
        msg = self._admin.reset_group(group_id)
        await self._state.save_dirty(self._kv_save)
        event.set_result(msg)

    @iris.command("cooldown")
    async def cmd_cooldown(self, event, minutes: int = 5) -> None:
        group_id = self._get_group_id(event)
        if not group_id:
            return
        msg = self._admin.set_cooldown(group_id, minutes)
        await self._state.save_dirty(self._kv_save)
        event.set_result(msg)

    @iris.command("willingness")
    async def cmd_willingness(self, event, level: str = "") -> None:
        group_id = self._get_group_id(event)
        if not group_id:
            return
        if not level.strip():
            current = self._admin.get_willingness(group_id)
            event.set_result(f"群 {group_id} 当前回复意愿: {current}\n可选: 低/中/高 (low/medium/high)")
            return
        msg = self._admin.set_willingness(group_id, level.strip())
        await self._state.save_dirty(self._kv_save)
        event.set_result(msg)

    @iris.command("initiate")
    async def cmd_initiate(self, event) -> None:
        group_id = self._get_group_id(event)
        if not group_id:
            return
        result = await self._proactive.attempt_initiate(group_id, force=True)
        event.set_result(f"主动发起: {result}")

    # ---- 消息唤醒：门控 → 标记 → 交由 on_llm_request 决策 ----

    @event_message_type(EventMessageType.GROUP_MESSAGE)
    async def on_message(self, event) -> None:
        if not self._config.enabled:
            return

        if not self._gatekeeper.should_process(event):
            return

        group_id = event.get_group_id()
        if not group_id:
            return

        # 缓存会话标识与自身 ID，供主动发起通路使用
        umo = getattr(event, "unified_msg_origin", "")
        if umo and self._group_umo.get(group_id) != umo:
            self._group_umo[group_id] = umo
            self._umo_dirty = True
        if not self._self_id:
            self._self_id = event.get_self_id() or ""

        message_str = event.message_str or ""
        sender_id = event.get_sender_id()
        sender_name = event.get_sender_name() or sender_id

        # 发起后的首次接话：清除 pending，该消息直接获得一次跟进评估资格
        pending_reply = self._state.consume_initiate_pending(group_id)
        is_followed = bool(sender_id and self._state.match_anchor_user(group_id, sender_id))

        score = self._gatekeeper.quality_score(message_str)
        if score < self._config.quality_threshold and not is_followed and not pending_reply:
            return

        self._sliding_window.append(
            group_id,
            WindowMessage(
                sender_id=sender_id,
                sender_name=sender_name,
                content=message_str,
                timestamp=time.time(),
            ),
        )

        if event.is_at_or_wake_command:
            self._triggering.pop(group_id, None)
            self._state.increment_msg_count(group_id)
            self._passive_active[group_id] = time.time()
            event.set_extra("iris_mode", "passive")
            return

        if self._is_busy(group_id) or self._proactive.is_initiating(group_id):
            logger.debug("Iris Reply: reply already in progress for group %s", group_id)
            return

        async with self._state.get_lock(group_id):
            motive = self._signals.evaluate_message(group_id, sender_id, message_str)

        if not motive and pending_reply:
            motive = "follow_up"
        if not motive:
            return

        is_follow_up = motive == "follow_up"

        if is_follow_up:
            if group_id in self._follow_pending:
                logger.debug("Iris Reply: follow-up aggregation pending for group %s", group_id)
                return
            self._follow_pending.add(group_id)
            try:
                await asyncio.sleep(self._config.follow_up_aggregate_window)
            finally:
                self._follow_pending.discard(group_id)

            if self._is_busy(group_id):
                return
            if not pending_reply and not self._state.get_anchor(group_id).active:
                return

        if not self._state.can_detect(group_id, follow_up=is_follow_up):
            logger.debug("Iris Reply: trigger rate-limited for group %s", group_id)
            return

        provider_id = self._config.provider_id
        if not provider_id:
            provider_id = await self._get_provider_id(event)
            if not provider_id:
                logger.error("Iris Reply: failed to get provider ID for group %s", group_id)
                return

        async with self._state.get_lock(group_id):
            if group_id in self._triggering:
                logger.debug("Iris Reply: trigger already in progress for group %s", group_id)
                return
            self._state.record_detect_time(group_id)
            self._triggering[group_id] = time.time()

        event.set_extra("iris_decision", {
            "motive": motive,
            "provider_id": provider_id,
        })

        event.is_at_or_wake_command = True
        event.is_wake = True
        if provider_id:
            event.set_extra("selected_provider", provider_id)
        self._tool_ctx.set_context(group_id)
        logger.info(
            "Iris Reply: %s candidate activated for group %s, deferred to on_llm_request",
            motive, group_id,
        )

    # ---- 统一决策：消息唤醒的 LLM 评估 ----

    @on_llm_request()
    async def handle_llm_request(self, event, request: ProviderRequest) -> None:
        group_id = event.get_group_id()
        if not group_id or group_id not in self._triggering:
            return

        info = event.get_extra("iris_decision")
        if not info:
            self._triggering.pop(group_id, None)
            return

        motive = info.get("motive", "")
        provider_id = info.get("provider_id", "")

        req = DecisionRequest(group_id=group_id, wake="message", motive=motive)
        outcome = await self._decision_core.decide(req, self.context.llm_generate, provider_id)

        if outcome.error or outcome.decision is None:
            logger.error("Iris Reply: decision LLM call failed for group %s: %s", group_id, outcome.error)
            self._stats.record_decision_error(group_id, motive)
            async with self._state.get_lock(group_id):
                self._state.record_skip_reply(group_id)
            await self._state.save_dirty(self._kv_save)
            self._triggering.pop(group_id, None)
            event.stop_event()
            return

        decision = outcome.decision
        logger.info(
            "Iris Reply: decision raw for group %s (motive=%s, len=%d): %.500s",
            group_id, motive, len(outcome.raw_text), outcome.raw_text,
        )
        self._stats.record_decision(
            group_id, motive,
            system_prompt=outcome.system_prompt,
            user_prompt=outcome.user_prompt,
            response_text=outcome.raw_text,
            decision=decision,
            duration_ms=outcome.duration_ms,
        )
        logger.info(
            "Iris Reply: decision parsed for group %s: speak=%s, drifted=%s, watch=%s, watch_keywords=%s, cooldown=%d",
            group_id, decision.should_speak, decision.drifted,
            decision.watch, decision.watch_keywords, decision.cooldown_minutes,
        )

        async with self._state.get_lock(group_id):
            if decision.observation:
                self._state.set_observation(group_id, decision.observation)

        if decision.parse_failed:
            logger.warning("Iris Reply: decision parse failed for group %s", group_id)
            async with self._state.get_lock(group_id):
                self._state.record_skip_reply(group_id)
            await self._state.save_dirty(self._kv_save)
            self._triggering.pop(group_id, None)
            event.stop_event()
            return

        if group_id in self._passive_active:
            logger.info("Iris Reply: aborting %s for group %s, passive reply in progress", motive, group_id)
            async with self._state.get_lock(group_id):
                self._state.record_skip_reply(group_id)
            await self._state.save_dirty(self._kv_save)
            self._triggering.pop(group_id, None)
            event.stop_event()
            return

        if decision.cooldown_minutes:
            async with self._state.get_lock(group_id):
                actual = self._state.set_cooldown(group_id, decision.cooldown_minutes)
                self._state.record_skip_reply(group_id)
            await self._state.save_dirty(self._kv_save)
            logger.info("Iris Reply: decision requested cooldown %d min for group %s", actual, group_id)
            self._triggering.pop(group_id, None)
            event.stop_event()
            return

        if decision.drifted:
            async with self._state.get_lock(group_id):
                self._state.close_anchor(group_id)
                self._state.record_drift(group_id)
            await self._state.save_dirty(self._kv_save)
            logger.info("Iris Reply: topic drifted for group %s, anchor closed", group_id)
            self._triggering.pop(group_id, None)
            event.stop_event()
            return

        if decision.watch or decision.watch_keywords:
            if decision.should_speak:
                event.set_extra("iris_pending_watch", (
                    decision.watch, decision.watch_keywords, decision.why,
                ))
            else:
                async with self._state.get_lock(group_id):
                    self._state.add_anchor_watch(
                        group_id,
                        users=decision.watch or None,
                        keywords=decision.watch_keywords or None,
                        reason=decision.why,
                    )
            logger.info(
                "Iris Reply: decision watch for group %s, users=%s, keywords=%s, reason=%s (speak=%s)",
                group_id, decision.watch, decision.watch_keywords, decision.why, decision.should_speak,
            )

        if not decision.should_speak:
            async with self._state.get_lock(group_id):
                self._state.record_skip_reply(group_id)
            await self._state.save_dirty(self._kv_save)
            logger.debug("Iris Reply: decision skip for group %s", group_id)
            self._triggering.pop(group_id, None)
            event.stop_event()
            return

        self._reply_in_progress[group_id] = time.time()
        self._triggering.pop(group_id, None)

        event.set_extra("iris_mode", motive)
        event.set_extra("iris_decision_obs", decision.observation)

        hint = SPEAK_HINTS.get(motive, SPEAK_HINTS["chime_in"])
        request.extra_user_content_parts.append(TextPart(text=hint).mark_as_temp())
        logger.info("Iris Reply: decision speak (%s) for group %s", motive, group_id)

    @on_llm_response()
    async def handle_llm_response(self, event, response: LLMResponse) -> None:
        group_id = event.get_group_id()
        if not group_id:
            return

        event.set_extra("iris_llm_replied", True)
        mode = event.get_extra("iris_mode")

        if mode in ("chime_in", "follow_up"):
            self._reply_in_progress.pop(group_id, None)
            self._passive_active.pop(group_id, None)

            async with self._state.get_lock(group_id):
                self._state.record_actual_reply(group_id)

            self._tool_ctx.clear_context()
            await self._state.save_dirty(self._kv_save)
            logger.info("Iris Reply: %s reply sent for group %s", mode, group_id)
        elif mode == "passive":
            self._passive_active.pop(group_id, None)

            async with self._state.get_lock(group_id):
                self._state.record_actual_reply(group_id, count_consecutive=False)

            self._stats.record_passive_reply(group_id)
            await self._state.save_dirty(self._kv_save)
            logger.info("Iris Reply: passive reply boost applied for group %s", group_id)
        else:
            if not self._state.is_whitelisted(group_id):
                return
            async with self._state.get_lock(group_id):
                self._state.record_actual_reply(group_id, count_consecutive=False)
            await self._state.save_dirty(self._kv_save)
            logger.info("Iris Reply: normal LLM reply for group %s, boost applied", group_id)

    @after_message_sent()
    async def on_message_sent(self, event) -> None:
        group_id = event.get_group_id()
        if not group_id:
            return
        if not self._state.is_whitelisted(group_id):
            return
        sender_id = event.get_sender_id()
        if not sender_id:
            return
        result = event.get_result()
        bot_text = result.get_plain_text().strip() if result else ""
        if bot_text:
            self._sliding_window.append(group_id, WindowMessage(
                sender_id=event.get_self_id() or "iris",
                sender_name="Iris",
                content=bot_text,
                timestamp=time.time(),
            ))
        mode = event.get_extra("iris_mode")
        if mode in ("chime_in", "follow_up"):
            pending = event.get_extra("iris_pending_watch")
            users = list(pending[0]) if pending else []
            if sender_id not in users:
                users.append(sender_id)
            keywords = list(pending[1]) if pending else []
            reason = pending[2] if pending else ""
            topic = event.get_extra("iris_decision_obs", "")
            async with self._state.get_lock(group_id):
                self._state.write_anchor(
                    group_id,
                    kind=mode,
                    topic=topic,
                    bot_message=bot_text,
                    users=users,
                    keywords=keywords or None,
                    reason=reason,
                )
            await self._state.save_dirty(self._kv_save)
            logger.debug("Iris Reply: anchor written (%s) for group %s", mode, group_id)
        elif mode == "passive":
            provider_id = self._config.provider_id or await self._get_provider_id(event)
            if provider_id:
                await self._passive_watch_eval(group_id, provider_id, sender_id, bot_text)
            else:
                async with self._state.get_lock(group_id):
                    self._state.write_anchor(
                        group_id, kind="passive", bot_message=bot_text, users=[sender_id],
                    )
                await self._state.save_dirty(self._kv_save)
        elif event.get_extra("iris_llm_replied"):
            async with self._state.get_lock(group_id):
                self._state.write_anchor(
                    group_id, kind="reply", bot_message=bot_text, users=[sender_id],
                )
            await self._state.save_dirty(self._kv_save)
            logger.debug("Iris Reply: anchor written (reply) for group %s", group_id)

    async def _passive_watch_eval(
        self, group_id: str, provider_id: str, fallback_sender: str, bot_text: str,
    ) -> None:
        """被动回复后的跟进评估（motive=watch）：只决定是否建立关注锚点。"""
        if not self._sliding_window.get_messages(group_id):
            return

        req = DecisionRequest(group_id=group_id, wake="message", motive="watch")
        outcome = await self._decision_core.decide(req, self.context.llm_generate, provider_id)

        if outcome.error or outcome.decision is None:
            logger.warning("Iris Reply: passive watch eval failed for group %s: %s", group_id, outcome.error)
            self._stats.record_decision_error(group_id, "watch")
            async with self._state.get_lock(group_id):
                self._state.write_anchor(
                    group_id, kind="passive", bot_message=bot_text, users=[fallback_sender],
                )
            await self._state.save_dirty(self._kv_save)
            return

        decision = outcome.decision
        self._stats.record_decision(
            group_id, "watch",
            system_prompt=outcome.system_prompt,
            user_prompt=outcome.user_prompt,
            response_text=outcome.raw_text,
            decision=decision,
            duration_ms=outcome.duration_ms,
        )
        logger.info(
            "Iris Reply: passive watch eval for group %s: watch=%s, keywords=%s, drifted=%s",
            group_id, decision.watch, decision.watch_keywords, decision.drifted,
        )

        async with self._state.get_lock(group_id):
            if decision.observation:
                self._state.set_observation(group_id, decision.observation)
            if decision.parse_failed:
                self._state.write_anchor(
                    group_id, kind="passive", bot_message=bot_text, users=[fallback_sender],
                )
            elif decision.drifted:
                self._state.close_anchor(group_id)
                self._state.record_drift(group_id)
                logger.info("Iris Reply: topic drifted (passive) for group %s, anchor closed", group_id)
            elif decision.watch or decision.watch_keywords:
                self._state.write_anchor(
                    group_id,
                    kind="passive",
                    bot_message=bot_text,
                    users=decision.watch or None,
                    keywords=decision.watch_keywords or None,
                    reason=decision.why,
                )
            else:
                self._state.write_anchor(
                    group_id, kind="passive", bot_message=bot_text, users=[fallback_sender],
                )
        await self._state.save_dirty(self._kv_save)
