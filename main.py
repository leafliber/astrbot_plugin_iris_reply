from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_plugin_dir = str(Path(__file__).parent)
if _plugin_dir not in sys.path:
    sys.path.insert(0, _plugin_dir)

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
from iris_reply.prompts import SUMMARY_SYSTEM_PROMPT, WILLINGNESS_PROMPTS
from iris_reply.state import GroupState, StateManager
from iris_reply.tools import ToolContext
from iris_reply.trigger import TriggerEngine


@dataclass
class SummaryResult:
    observation: str = ""
    noteworthy: bool = False
    noteworthy_users: list[str] = field(default_factory=list)
    noteworthy_reason: str = ""


@dataclass
class TriggerResult:
    should_reply: bool = False
    topic_drifted: bool = False
    follow_up_users: list[str] = field(default_factory=list)
    interest_reason: str = ""


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
        self._iris_context: dict[str, str] = {}
        self._iris_active: set[str] = set()
        self._summary_cache: dict[str, SummaryResult] = {}
        self._summarizing: set[str] = set()
        self._triggering: set[str] = set()
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
                self._sliding_window.cleanup(self._state.get_whitelist())
            except Exception as e:
                logger.warning("Iris Reply: periodic save error: %s", e)

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
            self._state.set_cooldown(group_id, minutes)
        logger.debug("Iris Reply: set_cooldown for group %s, %d min", group_id, minutes)
        return f"ok: cooldown set for {max(1, min(120, int(minutes)))} minutes"

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

        summary_provider_id = self._config.summary_provider_id
        summary_provider = await self._get_provider_id(event, summary_provider_id)
        if summary_provider:
            asyncio.create_task(self._run_summary(group_id, summary_provider))

        if event.is_at_or_wake_command:
            self._state.increment_msg_count(group_id)
            return

        async with self._state.get_lock(group_id):
            trigger_reason = self._trigger_engine.evaluate(event)

        if not trigger_reason:
            return

        if not self._state.can_detect(group_id):
            logger.debug("Iris Reply: trigger rate-limited for group %s", group_id)
            return

        if group_id in self._triggering:
            logger.debug("Iris Reply: trigger already in progress for group %s", group_id)
            return

        self._triggering.add(group_id)
        try:
            await self._trigger_reply(event, group_id, trigger_reason)
        except Exception as e:
            logger.error("Iris Reply: trigger error for group %s: %s", group_id, e, exc_info=True)
        finally:
            self._triggering.discard(group_id)

    async def _run_summary(self, group_id: str, provider_id: str) -> None:
        if not self._state.can_summarize(group_id, self._config.summary_min_interval):
            return

        if group_id in self._summarizing:
            return

        self._summarizing.add(group_id)
        try:
            messages = self._sliding_window.get_messages(group_id)
            context_text = self._context_packager.package(group_id, messages, "summary")

            try:
                response = await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=context_text,
                    system_prompt=SUMMARY_SYSTEM_PROMPT,
                )
            except Exception as e:
                logger.error("Iris Reply: summary LLM call failed for group %s: %s", group_id, e)
                return

            self._state.record_summary_time(group_id)

            completion = response.completion_text or ""
            result = self._parse_summary(completion)
            logger.info(
                "Iris Reply: summary for group %s: noteworthy=%s, users=%s, reason=%s, observation=%.100s",
                group_id, result.noteworthy, result.noteworthy_users, result.noteworthy_reason, result.observation,
            )

            self._summary_cache[group_id] = result

            async with self._state.get_lock(group_id):
                group_state = self._state.get_state(group_id)

                if result.noteworthy and result.noteworthy_users:
                    overlap = set(result.noteworthy_users) & group_state.follow_up.user_ids
                    if overlap:
                        self._state.reset_patience(group_id)

                    self._state.add_follow_up(
                        group_id,
                        user_ids=result.noteworthy_users,
                        reason=result.noteworthy_reason,
                    )
                elif result.noteworthy and group_state.state == GroupState.FOLLOWING:
                    self._state.reset_patience(group_id)
                elif not result.noteworthy and group_state.state == GroupState.FOLLOWING and len(messages) >= 3:
                    patience = self._state.increment_patience(group_id)
                    if patience >= self._config.follow_up_patience:
                        logger.info(
                            "Iris Reply: follow-up patience exhausted for group %s (%d), clearing",
                            group_id, patience,
                        )
                        self._state.clear_follow_up(group_id)

                await self._state.save_dirty(self._kv_save)
        except Exception as e:
            logger.error("Iris Reply: summary error for group %s: %s", group_id, e, exc_info=True)
        finally:
            self._summarizing.discard(group_id)

    async def _trigger_reply(
        self, event, group_id: str, trigger_reason: str
    ) -> None:
        self._state.record_detect_time(group_id)

        messages = self._sliding_window.get_messages(group_id)
        context_text = self._context_packager.package(group_id, messages, trigger_reason)

        provider_id = self._config.provider_id
        if not provider_id:
            provider_id = await self._get_provider_id(event)
            if not provider_id:
                logger.error("Iris Reply: failed to get provider ID for group %s", group_id)
                return

        willingness = self._state.get_willingness(group_id)
        prompts = WILLINGNESS_PROMPTS[willingness]
        user_prompt = prompts["persona"]

        cached_summary = self._summary_cache.get(group_id)
        if cached_summary:
            user_prompt += (
                f"\n\n<recent_observation>"
                f"观察概括：{cached_summary.observation}"
            )
            if cached_summary.noteworthy:
                user_prompt += f"\n关注原因：{cached_summary.noteworthy_reason}"
            user_prompt += "</recent_observation>"

        if trigger_reason == "follow_up":
            follow_up_users, follow_up_reason = self._state.get_follow_up_info(group_id)
            if follow_up_users:
                user_prompt += (
                    f"\n\n<follow_up_reminder>"
                    f"你之前表示对这些用户感兴趣：{', '.join(follow_up_users)}"
                )
                if follow_up_reason:
                    user_prompt += f"，原因：{follow_up_reason}"
                user_prompt += (
                    "。现在其中有人发言了，请认真考虑是否回复。"
                    "如果当前话题已经偏离了你之前关注的原因，将 topic_drifted 设为 true。"
                    "</follow_up_reminder>"
                )

        user_prompt += "\n\n" + context_text

        try:
            response = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=user_prompt,
                system_prompt=prompts["trigger_system"],
            )
        except Exception as e:
            logger.error("Iris Reply: trigger LLM call failed for group %s: %s", group_id, e)
            return

        completion = response.completion_text or ""
        logger.info("Iris Reply: trigger raw response for group %s (len=%d): %.500s", group_id, len(completion), completion)
        result = self._parse_trigger(completion)
        logger.info(
            "Iris Reply: trigger parsed for group %s: should_reply=%s, topic_drifted=%s, follow_up=%s",
            group_id, result.should_reply, result.topic_drifted, result.follow_up_users,
        )

        if result.topic_drifted:
            async with self._state.get_lock(group_id):
                self._state.clear_follow_up(group_id)
            await self._state.save_dirty(self._kv_save)
            logger.info("Iris Reply: topic drifted for group %s, cleared follow-up", group_id)
            return

        if result.follow_up_users:
            async with self._state.get_lock(group_id):
                self._state.add_follow_up(group_id, user_ids=result.follow_up_users, reason=result.interest_reason)
            logger.info(
                "Iris Reply: trigger follow-up for group %s, users=%s, reason=%s",
                group_id, result.follow_up_users, result.interest_reason,
            )

        if not result.should_reply:
            async with self._state.get_lock(group_id):
                self._state.record_skip_reply(group_id)
            await self._state.save_dirty(self._kv_save)
            logger.debug("Iris Reply: trigger skip for group %s", group_id)
            return

        summary_text = ""
        if cached_summary:
            summary_text = cached_summary.observation

        self._iris_context[group_id] = summary_text
        self._iris_active.add(group_id)
        event.is_at_or_wake_command = True
        event.is_wake = True
        if provider_id:
            event.set_extra("selected_provider", provider_id)
        self._tool_ctx.set_context(group_id)
        logger.info(
            "Iris Reply: trigger passed (%s) for group %s (is_at_or_wake_command=%s, is_wake=%s)",
            trigger_reason, group_id, event.is_at_or_wake_command, event.is_wake,
        )

    @on_llm_request()
    async def handle_llm_request(self, event, request: ProviderRequest) -> None:
        group_id = event.get_group_id()
        if not group_id or group_id not in self._iris_context:
            return

        summary_text = self._iris_context.pop(group_id, None)
        if summary_text is None:
            return

        willingness = self._state.get_willingness(group_id)
        persona = WILLINGNESS_PROMPTS[willingness]["persona"]

        messages = self._sliding_window.get_messages(group_id)
        context_text = self._context_packager.package(group_id, messages, "reply")

        combined = persona
        if summary_text:
            combined += f"\n\n<conversation_summary>{summary_text}</conversation_summary>"
        combined += "\n\n" + context_text

        request.extra_user_content_parts.append(TextPart(text=combined))

        logger.info("Iris Reply: injected context for group %s (willingness=%s)", group_id, willingness)

    @on_llm_response()
    async def handle_llm_response(self, event, response: LLMResponse) -> None:
        group_id = event.get_group_id()
        if not group_id or group_id not in self._iris_active:
            return

        self._iris_active.discard(group_id)

        async with self._state.get_lock(group_id):
            self._state.record_actual_reply(group_id)
            self._state.clear_follow_up(group_id)

        self._tool_ctx.clear_context()
        await self._state.save_dirty(self._kv_save)
        logger.info("Iris Reply: actual reply sent for group %s, follow-up cleared", group_id)

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

        text = re.sub(r'"\s*:\s*True', '": true', text)
        text = re.sub(r'"\s*:\s*False', '": false', text)

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
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
                            return obj
                    except json.JSONDecodeError:
                        start = -1

        return None

    @staticmethod
    def _parse_bool(val: Any) -> bool:
        if isinstance(val, str):
            return val.lower() in ("true", "yes", "1")
        if isinstance(val, (int, float)):
            return bool(val)
        return bool(val)

    @staticmethod
    def _parse_string_list(raw: Any, max_len: int = 10) -> list[str]:
        if not isinstance(raw, list):
            return []
        result = [str(u).strip() for u in raw if str(u).strip()]
        return result[:max_len]

    @classmethod
    def _parse_summary(cls, text: str) -> SummaryResult:
        obj = cls._extract_json(text)
        if not obj:
            logger.warning("Iris Reply: summary JSON parse failed, raw text: %.300s", text)
            return SummaryResult()

        return SummaryResult(
            observation=str(obj.get("observation", "")),
            noteworthy=cls._parse_bool(obj.get("noteworthy", False)),
            noteworthy_users=cls._parse_string_list(obj.get("noteworthy_users", [])),
            noteworthy_reason=str(obj.get("noteworthy_reason", "")),
        )

    @classmethod
    def _parse_trigger(cls, text: str) -> TriggerResult:
        obj = cls._extract_json(text)
        if not obj:
            logger.warning("Iris Reply: trigger JSON parse failed, raw text: %.300s", text)
            return TriggerResult()

        return TriggerResult(
            should_reply=cls._parse_bool(obj.get("should_reply", False)),
            topic_drifted=cls._parse_bool(obj.get("topic_drifted", False)),
            follow_up_users=cls._parse_string_list(obj.get("follow_up_users", [])),
            interest_reason=str(obj.get("interest_reason", "")),
        )
