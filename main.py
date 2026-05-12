from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from astrbot.api import logger
from astrbot.api.event.filter import (
    EventMessageType,
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

from iris_reply.admin import AdminCommands
from iris_reply.config import ConfigManager
from iris_reply.perception import ContextPackager, Gatekeeper, SlidingWindow, WindowMessage
from iris_reply.prompts import WILLINGNESS_PROMPTS
from iris_reply.state import StateManager
from iris_reply.tools import ToolContext
from iris_reply.trigger import TriggerEngine


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
        self._detecting: set[str] = set()
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

    @llm_tool(name="add_follow_up")
    async def tool_add_follow_up(self, event, user_ids: str = "") -> str:
        """当你希望持续关注某些用户的发言时调用此工具。Iris 将在后续消息中匹配指定用户时自动触发回复。

        Args:
            user_ids(string): 逗号分隔的用户ID列表，如 "user1,user2"
        """
        group_id = self._tool_ctx.current_group_id or event.get_group_id()
        if not group_id:
            return "error: no group context"

        uid_list = [u.strip() for u in user_ids.split(",") if u.strip()] if user_ids else None

        if not uid_list:
            return "error: must provide at least one user_id"

        async with self._state.get_lock(group_id):
            self._state.add_follow_up(group_id, user_ids=uid_list)
        logger.debug(f"Iris Reply: add_follow_up for group {group_id}, users={uid_list}")
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
        logger.debug(f"Iris Reply: end_follow_up for group {group_id}, users={uid_list}")
        return f"ok: removed follow-up users={uid_list}"

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

    @command_group("iris_reply")
    def iris(self):
        pass

    @iris.command("enable")
    @permission_type(PermissionType.ADMIN)
    @event_message_type(EventMessageType.GROUP_MESSAGE)
    async def cmd_enable(self, event) -> None:
        group_id = event.get_group_id()
        if not group_id:
            event.set_result("无法获取群ID")
            return
        self._state.add_to_whitelist(group_id)
        await self._state.save_all(self._kv_save)
        event.set_result(f"群 {group_id} 已启用 Iris Reply")

    @iris.command("disable")
    @permission_type(PermissionType.ADMIN)
    @event_message_type(EventMessageType.GROUP_MESSAGE)
    async def cmd_disable(self, event) -> None:
        group_id = event.get_group_id()
        if not group_id:
            event.set_result("无法获取群ID")
            return
        self._state.remove_from_whitelist(group_id)
        await self._state.save_all(self._kv_save)
        event.set_result(f"群 {group_id} 已禁用 Iris Reply")

    @iris.command("status")
    @permission_type(PermissionType.ADMIN)
    @event_message_type(EventMessageType.GROUP_MESSAGE)
    async def cmd_status(self, event) -> None:
        group_id = event.get_group_id()
        if not group_id:
            event.set_result("无法获取群ID")
            return
        text = self._admin.get_status(group_id)
        event.set_result(text)

    @iris.command("reset")
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

    @iris.command("cooldown")
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

    @iris.command("willingness")
    @permission_type(PermissionType.ADMIN)
    @event_message_type(EventMessageType.GROUP_MESSAGE)
    async def cmd_willingness(self, event, level: str = "") -> None:
        group_id = event.get_group_id()
        if not group_id:
            event.set_result("无法获取群ID")
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

        if not trigger_reason:
            return

        if group_id in self._detecting:
            logger.debug(f"Iris Reply: detection already in progress for group {group_id}")
            return

        self._detecting.add(group_id)
        try:
            await self._detect_and_trigger(event, group_id, trigger_reason)
        except Exception as e:
            logger.error(f"Iris Reply: detection error for group {group_id}: {e}", exc_info=True)
        finally:
            self._detecting.discard(group_id)

    async def _detect_and_trigger(
        self, event, group_id: str, trigger_reason: str
    ) -> None:
        messages = self._sliding_window.get_messages(group_id)
        context_text = self._context_packager.package(group_id, messages, trigger_reason)

        provider_id = self._config.provider_id
        if not provider_id:
            try:
                provider_id = await self.context.get_current_chat_provider_id(
                    event.unified_msg_origin
                )
            except Exception:
                logger.error(f"Iris Reply: failed to get provider ID for group {group_id}")
                return

        willingness = self._state.get_willingness(group_id)
        prompts = WILLINGNESS_PROMPTS[willingness]
        user_prompt = prompts["persona"] + "\n\n" + context_text

        try:
            response = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=user_prompt,
                system_prompt=prompts["detection_system"],
            )
        except Exception as e:
            logger.error(f"Iris Reply: detection LLM call failed for group {group_id}: {e}")
            return

        completion = response.completion_text or ""
        should_reply = self._parse_should_reply(completion)

        if not should_reply:
            async with self._state.get_lock(group_id):
                self._state.record_skip_reply(group_id)
            await self._state.save_dirty(self._kv_save)
            logger.debug(f"Iris Reply: detection skip for group {group_id}")
            return

        self._iris_context[group_id] = context_text
        self._iris_active.add(group_id)
        event.is_at_or_wake_command = True
        if provider_id:
            event.set_extra("selected_provider", provider_id)
        self._tool_ctx.set_context(group_id)
        logger.info(f"Iris Reply: detection passed ({trigger_reason}), triggering standard pipeline for group {group_id}")

    @on_llm_request()
    async def handle_llm_request(self, event, request: ProviderRequest) -> None:
        group_id = event.get_group_id()
        if not group_id or group_id not in self._iris_context:
            return

        context_text = self._iris_context.pop(group_id)
        willingness = self._state.get_willingness(group_id)
        persona = WILLINGNESS_PROMPTS[willingness]["persona"]
        combined = f"{persona}\n\n{context_text}"

        request.extra_user_content_parts.append(TextPart(text=combined))

        logger.debug(f"Iris Reply: injected context for group {group_id}")

    @on_llm_response()
    async def handle_llm_response(self, event, response: LLMResponse) -> None:
        group_id = event.get_group_id()
        if not group_id or group_id not in self._iris_active:
            return

        self._iris_active.discard(group_id)

        async with self._state.get_lock(group_id):
            self._state.record_actual_reply(group_id)

        self._tool_ctx.clear_context()
        await self._state.save_dirty(self._kv_save)
        logger.debug(f"Iris Reply: actual reply for group {group_id}")

    @staticmethod
    def _parse_should_reply(text: str) -> bool:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return bool(obj.get("should_reply", False))
        except json.JSONDecodeError:
            pass

        brace_count = 0
        start = -1
        for i, c in enumerate(text):
            if c == "{":
                if brace_count == 0:
                    start = i
                brace_count += 1
            elif c == "}":
                brace_count -= 1
                if brace_count == 0 and start >= 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return bool(obj.get("should_reply", False))
                    except json.JSONDecodeError:
                        start = -1

        return False
