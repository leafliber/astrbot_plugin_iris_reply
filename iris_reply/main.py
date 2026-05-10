from __future__ import annotations

import asyncio
import time
from typing import Any

from astrbot.api import logger
from astrbot.api.event.filter import (
    EventMessageType,
    command,
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

from .admin import AdminCommands
from .config import ConfigManager
from .perception import ContextPackager, Gatekeeper, SlidingWindow, WindowMessage
from .state import StateManager
from .tools import ToolContext
from .trigger import TriggerEngine


class IrisReply(Star):
    """Iris Reply - 信号驱动的群聊主动回复插件"""

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
        self._iris_context: dict[str, str] = {}
        self._iris_active: set[str] = set()
        self._save_task: asyncio.Task | None = None
        self._save_interval = 30

    async def initialize(self) -> None:
        await self._state.load_all(self._kv_load)
        self._save_task = asyncio.create_task(self._periodic_save())
        logger.info("Iris Reply: initialized")

    async def terminate(self) -> None:
        if self._save_task and not self._save_task.done():
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
        await self._state.save_all(self._kv_save)
        logger.info("Iris Reply: terminated")

    async def _periodic_save(self) -> None:
        while True:
            await asyncio.sleep(self._save_interval)
            try:
                await self._state.save_dirty(self._kv_save)
            except Exception as e:
                logger.warning(f"Iris Reply: periodic save error: {e}")

    async def _kv_save(self, key: str, value: Any) -> None:
        await self.put_kv_data(key, value)

    async def _kv_load(self, key: str) -> Any:
        return await self.get_kv_data(key, None)

    @llm_tool(name="skip_reply")
    async def tool_skip_reply(self, event) -> str:
        """当你认为不需要回复时调用此工具。调用后 Iris 将不发送任何消息，并自动增加退避倍率以减少后续触发频率。"""
        group_id = self._tool_ctx.current_group_id or event.get_group_id()
        if not group_id:
            return "error: no group context"
        async with self._state.get_lock(group_id):
            self._state.record_skip_reply(group_id)
        logger.debug(f"Iris Reply: skip_reply for group {group_id}")
        return "ok: skipped reply, backoff increased"

    @llm_tool(name="add_follow_up")
    async def tool_add_follow_up(self, event, user_ids: str = "", keywords: str = "") -> str:
        """当你希望持续关注某个话题或某些用户的发言时调用此工具。Iris 将在后续消息中匹配指定用户或关键字时自动触发回复。至少提供一个参数。

        Args:
            user_ids(string): 逗号分隔的用户ID列表，如 "user1,user2"
            keywords(string): 逗号分隔的关键字列表，如 "关键词1,关键词2"
        """
        group_id = self._tool_ctx.current_group_id or event.get_group_id()
        if not group_id:
            return "error: no group context"

        uid_list = [u.strip() for u in user_ids.split(",") if u.strip()] if user_ids else None
        kw_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else None

        if not uid_list and not kw_list:
            return "error: must provide at least one user_id or keyword"

        async with self._state.get_lock(group_id):
            self._state.add_follow_up(group_id, user_ids=uid_list, keywords=kw_list)
        logger.debug(f"Iris Reply: add_follow_up for group {group_id}, users={uid_list}, keywords={kw_list}")
        return f"ok: following users={uid_list}, keywords={kw_list}"

    @llm_tool(name="end_follow_up")
    async def tool_end_follow_up(self, event, user_ids: str = "", keywords: str = "") -> str:
        """当你不再需要关注某个话题或用户时调用此工具，移除对应的跟进记录。不提供参数则移除所有跟进记录。

        Args:
            user_ids(string): 逗号分隔的用户ID列表，如 "user1,user2"
            keywords(string): 逗号分隔的关键字列表，如 "关键词1,关键词2"
        """
        group_id = self._tool_ctx.current_group_id or event.get_group_id()
        if not group_id:
            return "error: no group context"

        uid_list = [u.strip() for u in user_ids.split(",") if u.strip()] if user_ids else None
        kw_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else None

        async with self._state.get_lock(group_id):
            self._state.remove_follow_up(group_id, user_ids=uid_list, keywords=kw_list)
        logger.debug(f"Iris Reply: end_follow_up for group {group_id}, users={uid_list}, keywords={kw_list}")
        return f"ok: removed follow-up users={uid_list}, keywords={kw_list}"

    @llm_tool(name="set_cooldown")
    async def tool_set_cooldown(self, event, minutes: int = 5) -> str:
        """当你认为 Iris 应该暂时停止主动回复时调用此工具。设置冷却时间，冷却期间 Iris 不会主动触发任何回复。

        Args:
            minutes(number): 冷却时间（分钟），范围 1-120，默认 5
        """
        group_id = self._tool_ctx.current_group_id or event.get_group_id()
        if not group_id:
            return "error: no group context"

        minutes = max(1, min(120, int(minutes)))

        async with self._state.get_lock(group_id):
            self._state.set_cooldown(group_id, minutes)
        logger.debug(f"Iris Reply: set_cooldown for group {group_id}, {minutes} min")
        return f"ok: cooldown set for {minutes} minutes"

    @command("iris_reply_status")
    @permission_type(PermissionType.ADMIN)
    @event_message_type(EventMessageType.GROUP_MESSAGE)
    async def cmd_status(self, event) -> None:
        group_id = event.get_group_id()
        if not group_id:
            event.set_result("无法获取群ID")
            return
        text = self._admin.get_status(group_id)
        event.set_result(text)

    @command("iris_reply_reset")
    @permission_type(PermissionType.ADMIN)
    @event_message_type(EventMessageType.GROUP_MESSAGE)
    async def cmd_reset(self, event) -> None:
        group_id = event.get_group_id()
        if not group_id:
            event.set_result("无法获取群ID")
            return
        msg = self._admin.reset_group(group_id)
        await self._state.save_dirty(self._kv_save)
        event.set_result(msg)

    @command("iris_reply_cooldown")
    @permission_type(PermissionType.ADMIN)
    @event_message_type(EventMessageType.GROUP_MESSAGE)
    async def cmd_cooldown(self, event, minutes: int = 5) -> None:
        group_id = event.get_group_id()
        if not group_id:
            event.set_result("无法获取群ID")
            return
        msg = self._admin.set_cooldown(group_id, minutes)
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
        score = self._gatekeeper.quality_score(message_str)
        if score < self._config.quality_threshold:
            return

        sender_id = event.get_sender_id()
        sender_name = event.get_sender_name() or sender_id

        self._sliding_window.append(
            group_id,
            WindowMessage(
                sender_id=sender_id,
                sender_name=sender_name,
                content=message_str,
                timestamp=time.time(),
            ),
        )

        async with self._state.get_lock(group_id):
            trigger_reason = self._trigger_engine.evaluate(event)

        if trigger_reason:
            messages = self._sliding_window.get_messages(group_id)
            context_text = self._context_packager.package(group_id, messages, trigger_reason)
            self._iris_context[group_id] = context_text
            self._iris_active.add(group_id)
            event.is_at_or_wake_command = True
            self._tool_ctx.set_context(group_id)
            logger.info(f"Iris Reply: triggered ({trigger_reason}) for group {group_id}")

    @on_llm_request()
    async def handle_llm_request(self, event, request: ProviderRequest) -> None:
        group_id = event.get_group_id()
        if not group_id or group_id not in self._iris_context:
            return

        context_text = self._iris_context.pop(group_id)
        prompt = self._config.system_prompt_template
        combined = f"{prompt}\n\n{context_text}"

        request.extra_user_content_parts.append(TextPart(text=combined))

        logger.debug(f"Iris Reply: injected context for group {group_id}")

    @on_llm_response()
    async def handle_llm_response(self, event, response: LLMResponse) -> None:
        group_id = event.get_group_id()
        if not group_id or group_id not in self._iris_active:
            return

        self._iris_active.discard(group_id)

        skip_detected = "skip_reply" in response.tools_call_name

        if skip_detected:
            event.stop_event()
            async with self._state.get_lock(group_id):
                self._state.record_skip_reply(group_id)
            logger.debug(f"Iris Reply: skip_reply detected for group {group_id}")
        else:
            async with self._state.get_lock(group_id):
                self._state.record_actual_reply(group_id)
            logger.debug(f"Iris Reply: actual reply for group {group_id}")

        self._tool_ctx.clear_context()
        await self._state.save_dirty(self._kv_save)
