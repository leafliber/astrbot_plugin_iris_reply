from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

plugin_root = Path(__file__).parent
if str(plugin_root) not in sys.path:
    sys.path.insert(0, str(plugin_root))

from astrbot.api.star import Star, Context
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

from iris_reply.config.config import ReplyConfig
from iris_reply.core.memory_api import MemoryAPI
from iris_reply.core.llm_client import LLMClient
from iris_reply.core.message_builder import MessageBuilder
from iris_reply.core.reply_sender import ReplySender
from iris_reply.storage.store import Store
from iris_reply.models.models import FollowupPlan, KeywordSource
from iris_reply.utils.cooldown import CooldownManager
from iris_reply.utils.rate_limiter import RateLimiter
from iris_reply.keyword.keyword_store import KeywordStore
from iris_reply.keyword.keyword_matcher import KeywordMatcher
from iris_reply.keyword.keyword_validator import KeywordValidator
from iris_reply.keyword.keyword_generator import KeywordGenerator
from iris_reply.proactive.analyzer import MessageAnalyzer
from iris_reply.proactive.decision_engine import DecisionEngine
from iris_reply.proactive.context_assembler import ContextAssembler
from iris_reply.followup.followup_planner import FollowupPlanner
from iris_reply.followup.followup_scheduler import FollowupScheduler
from iris_reply.followup.followup_executor import FollowupExecutor
from iris_reply.commands.handler import CommandHandler

logger = logging.getLogger("iris_reply")


class IrisReply(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self._config = ReplyConfig(context)

        data_dir = os.path.join(get_astrbot_plugin_data_path(), "iris_reply")
        os.makedirs(data_dir, exist_ok=True)

        self._store = Store(data_dir)
        self._memory_api = MemoryAPI()
        self._llm_client = LLMClient(context)
        self._message_builder = MessageBuilder()
        self._reply_sender = ReplySender(context, self._memory_api)

        self._cooldown = CooldownManager(
            default_seconds=self._config.cooldown.get("default_seconds", 300)
        )
        self._rate_limiter = RateLimiter(
            max_per_hour=self._config.proactive.get("max_replies_per_hour", 3)
        )

        self._keyword_store = KeywordStore(
            self._store,
            global_static_keywords=self._config.keyword.get("static_keywords", []),
        )
        self._keyword_matcher = KeywordMatcher()
        self._keyword_validator = KeywordValidator(
            self._llm_client, self._message_builder, self._memory_api, self._config
        )
        self._keyword_generator: KeywordGenerator | None = None
        if self._config.keyword.get("dynamic_generation", True):
            self._keyword_generator = KeywordGenerator(
                self._llm_client, self._message_builder, self._memory_api, self._config
            )

        self._analyzer = MessageAnalyzer(self._llm_client, self._memory_api, self._config)
        self._decision_engine = DecisionEngine(
            self._llm_client, self._message_builder, self._memory_api, self._config
        )
        self._context_assembler = ContextAssembler(self._memory_api, self._config)

        self._followup_planner = FollowupPlanner(
            self._llm_client, self._message_builder, self._memory_api, self._config
        )
        self._followup_scheduler = FollowupScheduler(self._store, self._config)
        self._followup_executor = FollowupExecutor(
            self._llm_client, self._message_builder, self._memory_api,
            self._reply_sender, self._cooldown, self._config,
        )

        self._followup_scheduler.set_execute_callback(self._execute_followup)

        self._command_handler = CommandHandler(
            context, self._store, self._keyword_store, self._keyword_generator
        )

        self._last_bot_reply: dict[str, str] = {}
        self._keyword_refresh_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        self._bind_tier_memory()
        self._load_group_cooldowns()

        if self._config.keyword.get("dynamic_generation", True) and self._keyword_generator:
            interval = self._config.keyword.get("dynamic_refresh_interval_minutes", 60)
            self._keyword_refresh_task = asyncio.create_task(
                self._periodic_keyword_refresh(interval)
            )

        logger.info("Iris Reply 初始化完成")

    async def terminate(self) -> None:
        if self._keyword_refresh_task and not self._keyword_refresh_task.done():
            self._keyword_refresh_task.cancel()
            try:
                await self._keyword_refresh_task
            except asyncio.CancelledError:
                pass

        await self._followup_scheduler.shutdown()
        self._store.close()
        logger.info("Iris Reply 已终止")

    def _bind_tier_memory(self) -> None:
        integration_mode = self._config.memory.get("integration_mode", "auto")
        if integration_mode == "off":
            logger.info("Memory 集成已关闭")
            return

        try:
            from iris_memory.core.components import get_component_manager
            cm = get_component_manager()
            if cm is not None:
                self._memory_api.bind(cm)
                logger.info("已绑定 tier_memory ComponentManager")
            else:
                logger.info("tier_memory ComponentManager 不可用，降级运行")
        except ImportError:
            logger.info("iris_memory 未安装，降级运行")
        except Exception as e:
            logger.warning("绑定 tier_memory 失败: %s，降级运行", e)

    def _load_group_cooldowns(self) -> None:
        groups = self._store.get_all_enabled_groups()
        for g in groups:
            self._cooldown.set_group_cooldown(g.group_id, g.cooldown_seconds)

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        group_id = self._get_group_id(event)
        if not group_id:
            return

        self._reply_sender.register_session(group_id, event.unified_msg_origin)

        user_id = self._get_user_id(event)
        message = event.message_str

        if not message:
            return

        if self._is_bot_message(event):
            return

        self._followup_scheduler.on_new_message(group_id)

        keyword_enabled = self._config.keyword.get("enable", True)
        if keyword_enabled:
            force_triggered = await self._handle_keyword(event, group_id, user_id, message)
            if force_triggered:
                return

        if getattr(event, "is_at_or_wake_command", False):
            return

        proactive_enabled = self._config.proactive.get("enable", False)
        group_config = self._store.get_group_config(group_id)
        if proactive_enabled and group_config.proactive_enabled:
            await self._handle_proactive(event, group_id, user_id, message)
        yield

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, response: Any) -> None:
        group_id = self._get_group_id(event)
        if not group_id:
            return

        followup_enabled = self._config.followup.get("enable", False)
        if not followup_enabled:
            return

        group_config = self._store.get_group_config(group_id)
        if not group_config.followup_enabled:
            return

        resp_text = ""
        if response is not None:
            if hasattr(response, "completion_text"):
                resp_text = str(response.completion_text).strip()
            elif hasattr(response, "completion"):
                resp_text = str(response.completion).strip()
            elif isinstance(response, str):
                resp_text = response.strip()
        if not resp_text:
            return

        self._last_bot_reply[group_id] = resp_text

        follow_all = self._config.followup.get("followup_after_all_replies", False)
        if not follow_all:
            return

        asyncio.create_task(self._plan_followup(group_id, resp_text))

    @filter.command("iris_reply")
    async def handle_command(self, event: AstrMessageEvent) -> None:
        message = event.message_str.strip()
        parts = message.split()
        args = parts[1:] if len(parts) > 1 else []
        result = await self._command_handler.handle(event, args)
        yield event.plain_result(result)

    async def _handle_keyword(
        self, event: AstrMessageEvent, group_id: str, user_id: str, message: str
    ) -> bool:
        if self._cooldown.is_on_cooldown(group_id):
            return False

        force_keywords = self._config.keyword.get("force_reply_keywords", [])
        if force_keywords:
            message_lower = message.lower()
            for fk in force_keywords:
                if fk.lower() in message_lower:
                    logger.info("命中强制回复关键词: %s，走 AstrBot 原生回复流程", fk)
                    event.is_at_or_wake_command = True
                    return True

        static_keywords = self._keyword_store.get_static_keywords(group_id)
        dynamic_keywords = self._keyword_store.get_dynamic_keywords(group_id)
        if not static_keywords and not dynamic_keywords:
            return False

        matches = []
        if static_keywords:
            matches.extend(self._keyword_matcher.match(message, static_keywords, KeywordSource.STATIC))
        if dynamic_keywords:
            matches.extend(self._keyword_matcher.match(message, dynamic_keywords, KeywordSource.DYNAMIC))
        if not matches:
            return False

        matches.sort(key=lambda m: m.confidence, reverse=True)
        best_match = matches[0]

        validation = await self._keyword_validator.validate(
            best_match, message, group_id, user_id
        )

        if not validation.should_reply:
            logger.debug(
                "关键词 '%s' 命中但验证不通过: %s",
                best_match.keyword, validation.reason,
            )
            return False

        direction = validation.reply_direction or f"用户消息涉及 '{best_match.keyword}'"
        context = await self._context_assembler.assemble(group_id, user_id, best_match.keyword)

        system_prompt, prompt = self._message_builder.build_keyword_reply_prompt(
            best_match, message, context, direction
        )
        provider_id = self._config.keyword.get("llm_provider_id", "")
        reply = await self._llm_client.generate(
            prompt, system_prompt, provider_id, module="keyword_reply"
        )

        if not reply:
            return False

        success = await self._reply_sender.send_group_message(group_id, reply)
        if success:
            self._cooldown.mark_reply(group_id)
            self._store.record_reply(
                group_id, "keyword", best_match.keyword, validation.confidence
            )
            self._last_bot_reply[group_id] = reply
            await self._plan_followup(group_id, reply)
            return True

        return False

    async def _handle_proactive(
        self, event: AstrMessageEvent, group_id: str, user_id: str, message: str
    ) -> None:
        if self._cooldown.is_on_cooldown(group_id):
            return

        if not self._rate_limiter.is_allowed(group_id):
            return

        recent = await self._memory_api.get_recent_context(
            group_id, self._config.proactive.get("analysis_window_messages", 10)
        )
        if not recent:
            current_msg = {
                "role": "user",
                "content": message,
                "sender_id": user_id,
            }
            recent = [current_msg]

        analysis = await self._analyzer.analyze(recent, group_id)

        if not analysis.should_consider:
            return

        decision = await self._decision_engine.decide(analysis, group_id, user_id)

        if not decision.should_reply:
            return

        direction = decision.reply_direction or analysis.summary
        reply = await self._decision_engine.generate_reply(
            analysis, group_id, user_id, direction
        )

        if not reply:
            return

        success = await self._reply_sender.send_group_message(group_id, reply)
        if success:
            self._cooldown.mark_reply(group_id)
            self._rate_limiter.record(group_id)
            self._store.record_reply(
                group_id, "proactive", None, decision.confidence
            )
            self._last_bot_reply[group_id] = reply

            followup_enabled = self._config.followup.get("enable", False)
            group_config = self._store.get_group_config(group_id)
            if followup_enabled and group_config.followup_enabled:
                await self._plan_followup(group_id, reply)

    async def _plan_followup(self, group_id: str, bot_reply: str) -> None:
        plan = await self._followup_planner.plan(group_id, bot_reply)
        if plan is None or not plan.should_followup:
            return

        await self._followup_scheduler.schedule(group_id, plan)

    async def _execute_followup(self, group_id: str, plan: FollowupPlan) -> None:
        bot_reply = self._last_bot_reply.get(group_id, "")
        await self._followup_executor.execute(group_id, plan, bot_reply)

    async def _periodic_keyword_refresh(self, interval_minutes: int) -> None:
        while True:
            try:
                await asyncio.sleep(interval_minutes * 60)
                groups = self._store.get_all_enabled_groups()
                for group in groups:
                    try:
                        existing = self._keyword_store.get_all_keywords(group.group_id)
                        keywords = await self._keyword_generator.refresh_all(
                            group.group_id, existing
                        )
                        self._keyword_store.update_dynamic_keywords(group.group_id, keywords)
                    except Exception as e:
                        logger.error("群 %s 动态关键词刷新失败: %s", group.group_id, e)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("动态关键词刷新任务异常: %s", e)

    def _get_group_id(self, event: AstrMessageEvent) -> str:
        try:
            gid = event.get_group_id()
            if gid:
                return str(gid)
        except Exception:
            pass
        return ""

    def _get_user_id(self, event: AstrMessageEvent) -> str:
        try:
            uid = event.get_sender_id()
            if uid:
                return str(uid)
        except Exception:
            pass
        return ""

    def _is_bot_message(self, event: AstrMessageEvent) -> bool:
        try:
            self_id = event.get_self_id()
            sender_id = event.get_sender_id()
            if self_id and sender_id and str(self_id) == str(sender_id):
                return True
        except Exception:
            pass
        return False
