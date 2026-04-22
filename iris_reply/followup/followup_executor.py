from __future__ import annotations

import logging
from typing import Any

from iris_reply.core.llm_client import LLMClient
from iris_reply.core.memory_api import MemoryAPI
from iris_reply.core.message_builder import MessageBuilder
from iris_reply.core.reply_sender import ReplySender
from iris_reply.models.models import (
    AssembledContext,
    FollowupPlan,
)
from iris_reply.config.config import ReplyConfig
from iris_reply.utils.cooldown import CooldownManager

logger = logging.getLogger("iris_reply.followup_executor")


class FollowupExecutor:
    def __init__(
        self,
        llm_client: LLMClient,
        message_builder: MessageBuilder,
        memory_api: MemoryAPI,
        reply_sender: ReplySender,
        cooldown_manager: CooldownManager,
        config: ReplyConfig,
    ):
        self._llm = llm_client
        self._builder = message_builder
        self._memory = memory_api
        self._sender = reply_sender
        self._cooldown = cooldown_manager
        self._config = config

    async def execute(self, group_id: str, plan: FollowupPlan, bot_reply: str) -> None:
        if self._cooldown.is_on_cooldown(group_id):
            remaining = self._cooldown.get_remaining(group_id)
            logger.info("群 %s 冷却中，跳过跟进（剩余 %.0f 秒）", group_id, remaining)
            return

        context = await self._assemble_context(group_id, plan.direction)
        system_prompt, prompt = self._builder.build_followup_reply_prompt(
            plan, bot_reply, context
        )
        provider_id = self._config.followup.get("llm_provider_id", "")
        reply = await self._llm.generate(
            prompt, system_prompt, provider_id, module="followup_reply"
        )

        if not reply:
            logger.warning("群 %s 跟进回复生成失败", group_id)
            return

        success = await self._sender.send_group_message(group_id, reply)
        if success:
            self._cooldown.mark_reply(group_id)
            logger.info("群 %s 跟进回复已发送", group_id)

    async def _assemble_context(self, group_id: str, query: str) -> AssembledContext:
        max_ctx = self._config.memory.get("max_context_messages", 10)
        max_mem = self._config.memory.get("max_memory_results", 5)
        max_kg = self._config.memory.get("max_knowledge_facts", 3)

        recent = await self._memory.get_recent_context(group_id, max_ctx)
        memories = await self._memory.search_memory(query, group_id, max_mem)
        facts = await self._memory.query_knowledge_graph(query, group_id)
        facts = facts[:max_kg]
        group_profile = await self._memory.get_group_profile(group_id)

        return AssembledContext(
            recent_messages=recent,
            relevant_memories=memories,
            knowledge_facts=facts,
            group_profile=group_profile,
        )
