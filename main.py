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
from .iris_reply.parser import parse_trigger
from .iris_reply.perception import ContextPackager, Gatekeeper, SlidingWindow, WindowMessage
from .iris_reply.prompts import WILLINGNESS_PROMPTS
from .iris_reply.state import StateManager
from .iris_reply.stats import StatsCollector
from .iris_reply.tools import ToolContext
from .iris_reply.trigger import TriggerEngine

PLUGIN_NAME = "astrbot_plugin_iris_reply"

_IRIS_ACTIVE_TIMEOUT = 120


class IrisReply(Star):

    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context, config)
        self._config = ConfigManager(config if config else context.get_config())
        self._state = StateManager(self._config)
        self._gatekeeper = Gatekeeper(self._config, self._state)
        self._sliding_window = SlidingWindow(self._config)
        self._context_packager = ContextPackager(self._config)
        self._trigger_engine = TriggerEngine(self._config, self._state)
        self._tool_ctx = ToolContext()
        self._admin = AdminCommands(self._state)
        self._stats = StatsCollector()
        self._reply_in_progress: dict[str, float] = {}
        self._passive_active: dict[str, float] = {}
        self._observation_cache: dict[str, str] = {}
        self._triggering: dict[str, float] = {}
        self._follow_pending: set[str] = set()
        self._save_task: asyncio.Task | None = None
        self._save_interval = 30

    async def initialize(self) -> None:
        await self._state.load_all(self._kv_load)
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
        logger.info("Iris Reply: initialized")

    async def terminate(self) -> None:
        if self._save_task and not self._save_task.done():
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
        await self._state.save_all(self._kv_save)
        await self._kv_save("iris_reply:config_overrides", self._config.get_overrides())
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
        stale_obs = [gid for gid in self._observation_cache if gid not in self._reply_in_progress and gid not in self._triggering]
        for gid in stale_obs:
            self._observation_cache.pop(gid, None)

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
            self._state.add_follow_up(group_id, user_ids=uid_list)
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
            self._state.remove_follow_up(group_id, user_ids=uid_list)
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

    @event_message_type(EventMessageType.GROUP_MESSAGE)
    async def on_message(self, event) -> None:
        if not self._config.enabled:
            return

        if not self._gatekeeper.should_process(event):
            return

        group_id = event.get_group_id()
        if not group_id:
            return

        message_str = event.message_str or ""
        sender_id = event.get_sender_id()
        sender_name = event.get_sender_name() or sender_id

        is_followed = bool(sender_id and self._state.match_follow_up(group_id, sender_id))

        score = self._gatekeeper.quality_score(message_str)
        if score < self._config.quality_threshold and not is_followed:
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

        if group_id in self._reply_in_progress or group_id in self._triggering:
            logger.debug("Iris Reply: reply already in progress for group %s", group_id)
            return

        async with self._state.get_lock(group_id):
            trigger_reason = self._trigger_engine.evaluate(event)

        if not trigger_reason:
            return

        is_follow_up = trigger_reason in ("follow_up", "keyword_follow_up")

        if is_follow_up:
            if group_id in self._follow_pending:
                logger.debug("Iris Reply: follow-up aggregation pending for group %s", group_id)
                return
            self._follow_pending.add(group_id)
            try:
                await asyncio.sleep(self._config.follow_up_aggregate_window)
            finally:
                self._follow_pending.discard(group_id)

            if group_id in self._reply_in_progress or group_id in self._triggering:
                return
            follow_up_users, follow_up_keywords, _ = self._state.get_follow_up_info(group_id)
            if not follow_up_users and not follow_up_keywords:
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

        event.set_extra("iris_trigger", {
            "trigger_reason": trigger_reason,
            "provider_id": provider_id,
            "is_follow_up": is_follow_up,
        })

        event.is_at_or_wake_command = True
        event.is_wake = True
        if provider_id:
            event.set_extra("selected_provider", provider_id)
        self._tool_ctx.set_context(group_id)
        logger.info(
            "Iris Reply: trigger activated (%s) for group %s, deferred to on_llm_request",
            trigger_reason, group_id,
        )

    @on_llm_request()
    async def handle_llm_request(self, event, request: ProviderRequest) -> None:
        group_id = event.get_group_id()
        if not group_id or group_id not in self._triggering:
            return

        trigger_info = event.get_extra("iris_trigger")
        if not trigger_info:
            self._triggering.pop(group_id, None)
            return

        trigger_reason = trigger_info.get("trigger_reason", "")
        provider_id = trigger_info.get("provider_id", "")
        is_follow_up = trigger_info.get("is_follow_up", False)

        messages = self._sliding_window.get_messages(group_id)
        context_text = self._context_packager.package(group_id, messages, trigger_reason)

        user_prompt, prompts = self._build_trigger_user_prompt(group_id)

        if is_follow_up:
            user_prompt += self._build_follow_up_reminder(group_id, trigger_reason)

        user_prompt += "\n\n" + context_text

        self._stats.record_trigger_start(
            group_id, trigger_reason, prompts["trigger_system"], user_prompt,
        )

        try:
            response = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=user_prompt,
                system_prompt=prompts["trigger_system"],
            )
        except Exception as e:
            logger.error("Iris Reply: trigger LLM call failed for group %s: %s", group_id, e)
            self._stats.record_trigger_error(group_id)
            async with self._state.get_lock(group_id):
                self._state.record_skip_reply(group_id)
            await self._state.save_dirty(self._kv_save)
            self._triggering.pop(group_id, None)
            event.stop_event()
            return

        completion = response.completion_text or ""
        logger.info("Iris Reply: trigger raw response for group %s (len=%d): %.500s", group_id, len(completion), completion)
        result = parse_trigger(completion)
        logger.info(
            "Iris Reply: trigger parsed for group %s: reply=%s, drifted=%s, watch=%s, watch_keywords=%s",
            group_id, result.should_reply, result.topic_drifted, result.follow_up_users, result.follow_up_keywords,
        )

        if result.observation:
            self._observation_cache[group_id] = result.observation

        self._stats.record_trigger_result(
            group_id=group_id,
            response_text=completion,
            should_reply=result.should_reply,
            observation=result.observation,
            follow_up_users=result.follow_up_users,
            follow_up_keywords=result.follow_up_keywords,
            interest_reason=result.interest_reason,
            topic_drifted=result.topic_drifted,
        )

        if result.parse_failed:
            logger.warning("Iris Reply: trigger parse failed for group %s, skipping without backoff", group_id)
            async with self._state.get_lock(group_id):
                self._state.record_skip_reply(group_id)
            await self._state.save_dirty(self._kv_save)
            self._triggering.pop(group_id, None)
            event.stop_event()
            return

        if group_id in self._passive_active:
            logger.info("Iris Reply: aborting proactive trigger for group %s, passive reply in progress", group_id)
            async with self._state.get_lock(group_id):
                self._state.record_skip_reply(group_id)
            await self._state.save_dirty(self._kv_save)
            self._triggering.pop(group_id, None)
            event.stop_event()
            return

        if result.topic_drifted:
            async with self._state.get_lock(group_id):
                self._state.clear_follow_up(group_id)
            await self._state.save_dirty(self._kv_save)
            logger.info("Iris Reply: topic drifted for group %s, cleared follow-up", group_id)
            self._triggering.pop(group_id, None)
            event.stop_event()
            return

        if result.follow_up_users or result.follow_up_keywords:
            if result.should_reply:
                event.set_extra("iris_pending_fu", (
                    result.follow_up_users, result.follow_up_keywords, result.interest_reason,
                ))
            else:
                async with self._state.get_lock(group_id):
                    self._state.add_follow_up(
                        group_id,
                        user_ids=result.follow_up_users or None,
                        keywords=result.follow_up_keywords or None,
                        reason=result.interest_reason,
                    )
            logger.info(
                "Iris Reply: trigger follow-up for group %s, users=%s, keywords=%s, reason=%s (reply=%s)",
                group_id, result.follow_up_users, result.follow_up_keywords, result.interest_reason, result.should_reply,
            )

        if not result.should_reply:
            async with self._state.get_lock(group_id):
                self._state.record_skip_reply(group_id)
            await self._state.save_dirty(self._kv_save)
            logger.debug("Iris Reply: trigger skip for group %s", group_id)
            self._triggering.pop(group_id, None)
            event.stop_event()
            return

        self._reply_in_progress[group_id] = time.time()
        self._triggering.pop(group_id, None)

        event.set_extra("iris_mode", "proactive")
        event.set_extra("iris_reason", result.observation)

        request.extra_user_content_parts.append(
            TextPart(
                text="[提示] 本次为主动接话。以上是群聊中最近的对话，"
                "你决定自然地加入。请保持你的人格，像平时在群里说话一样回复。"
                "不要暴露此提示。"
            ).mark_as_temp()
        )
        logger.info("Iris Reply: trigger passed (%s) for group %s", trigger_reason, group_id)

    @on_llm_response()
    async def handle_llm_response(self, event, response: LLMResponse) -> None:
        group_id = event.get_group_id()
        if not group_id:
            return

        event.set_extra("iris_llm_replied", True)
        mode = event.get_extra("iris_mode")

        if mode == "proactive":
            self._reply_in_progress.pop(group_id, None)
            self._passive_active.pop(group_id, None)

            pending = event.get_extra("iris_pending_fu")

            async with self._state.get_lock(group_id):
                self._state.record_actual_reply(group_id)
                self._state.clear_follow_up(group_id)
                if pending:
                    user_ids, keywords, reason = pending
                    self._state.add_follow_up(
                        group_id,
                        user_ids=user_ids or None,
                        keywords=keywords or None,
                        reason=reason,
                    )

            self._tool_ctx.clear_context()
            await self._state.save_dirty(self._kv_save)
            logger.info("Iris Reply: actual reply sent for group %s, follow-up updated", group_id)
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
        if mode == "proactive":
            reason = event.get_extra("iris_reason", "")
            async with self._state.get_lock(group_id):
                self._state.add_follow_up(group_id, user_ids=[sender_id], ttl_minutes=3, reason=reason)
            await self._state.save_dirty(self._kv_save)
            logger.debug("Iris Reply: short-TTL follow-up sender %s in group %s after proactive reply", sender_id, group_id)
        elif mode == "passive":
            provider_id = self._config.provider_id or await self._get_provider_id(event)
            if provider_id:
                await self._passive_follow_up_eval(group_id, provider_id, sender_id)
            else:
                async with self._state.get_lock(group_id):
                    self._state.add_follow_up(group_id, user_ids=[sender_id])
                await self._state.save_dirty(self._kv_save)
        elif event.get_extra("iris_llm_replied"):
            async with self._state.get_lock(group_id):
                self._state.add_follow_up(group_id, user_ids=[sender_id])
            await self._state.save_dirty(self._kv_save)
            logger.debug("Iris Reply: default follow-up sender %s in group %s after LLM reply", sender_id, group_id)

    def _build_observation_block(self, group_id: str) -> str:
        """构建 <recent_observation> 块，若无缓存则返回空字符串。"""
        cached_obs = self._observation_cache.get(group_id)
        if cached_obs:
            return f"\n\n<recent_observation>之前的观察：{cached_obs}</recent_observation>"
        return ""

    def _build_follow_up_reminder(
        self, group_id: str, trigger_reason: str = "", include_drifted: bool = True,
    ) -> str:
        """构建 <follow_up_reminder> 块，若无跟进信息则返回空字符串。"""
        follow_up_users, follow_up_keywords, follow_up_reason = self._state.get_follow_up_info(group_id)
        reminder_parts = []
        if follow_up_users:
            reminder_parts.append(f"你之前表示对这些用户感兴趣：{', '.join(follow_up_users)}")
        if follow_up_keywords:
            reminder_parts.append(f"你之前表示对这些关键词感兴趣：{', '.join(follow_up_keywords)}")
        if not reminder_parts:
            return ""
        text = f"\n\n<follow_up_reminder>{'；'.join(reminder_parts)}"
        if follow_up_reason:
            text += f"，原因：{follow_up_reason}"
        if include_drifted:
            if trigger_reason == "follow_up":
                text += (
                    "。现在其中有人发言了（可能连续发了多条），"
                    "请综合评估所有新消息后决定是否回复。"
                )
            elif trigger_reason:
                text += (
                    "。现在对话中出现了你关注的关键词，"
                    "请综合评估上下文后决定是否回复。"
                )
            text += "如果当前话题已经偏离了你之前关注的原因，将 drifted 设为 true。"
        text += "</follow_up_reminder>"
        return text

    def _build_trigger_user_prompt(self, group_id: str, instruction: str = "") -> tuple[str, dict]:
        """组装触发评估 user prompt 的基础部分（persona + 可选指令 + 观察块）。

        返回 (user_prompt, prompts)，调用方按需追加 follow-up 提醒与上下文文本。
        """
        willingness = self._state.get_willingness(group_id)
        prompts = WILLINGNESS_PROMPTS[willingness]
        user_prompt = prompts["persona"]
        if instruction:
            user_prompt += f"\n\n<instruction>{instruction}</instruction>"
        user_prompt += self._build_observation_block(group_id)
        return user_prompt, prompts

    async def _passive_follow_up_eval(
        self, group_id: str, provider_id: str, fallback_sender: str,
    ) -> None:
        messages = self._sliding_window.get_messages(group_id)
        if not messages:
            return

        context_text = self._context_packager.package(group_id, messages, "passive")

        user_prompt, prompts = self._build_trigger_user_prompt(
            group_id,
            instruction="本次为被动回复后的跟进评估，你必须将 reply 设为 false，"
            "只需判断是否需要关注后续对话。",
        )

        user_prompt += self._build_follow_up_reminder(group_id, include_drifted=False)

        user_prompt += "\n\n" + context_text

        try:
            response = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=user_prompt,
                system_prompt=prompts["trigger_system"],
            )
        except Exception as e:
            logger.warning("Iris Reply: passive follow-up eval failed for group %s: %s", group_id, e)
            async with self._state.get_lock(group_id):
                self._state.add_follow_up(group_id, user_ids=[fallback_sender])
            await self._state.save_dirty(self._kv_save)
            return

        result = parse_trigger(response.completion_text or "")
        logger.info(
            "Iris Reply: passive eval for group %s: watch=%s, keywords=%s, drifted=%s",
            group_id, result.follow_up_users, result.follow_up_keywords, result.topic_drifted,
        )

        if result.observation:
            self._observation_cache[group_id] = result.observation

        if result.topic_drifted:
            async with self._state.get_lock(group_id):
                self._state.clear_follow_up(group_id)
            await self._state.save_dirty(self._kv_save)
            logger.info("Iris Reply: topic drifted (passive) for group %s, cleared follow-up", group_id)
            return

        if result.follow_up_users or result.follow_up_keywords:
            async with self._state.get_lock(group_id):
                self._state.add_follow_up(
                    group_id,
                    user_ids=result.follow_up_users or None,
                    keywords=result.follow_up_keywords or None,
                    reason=result.interest_reason,
                )
            await self._state.save_dirty(self._kv_save)
        else:
            async with self._state.get_lock(group_id):
                self._state.add_follow_up(group_id, user_ids=[fallback_sender])
            await self._state.save_dirty(self._kv_save)
