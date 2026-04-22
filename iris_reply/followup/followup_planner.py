from __future__ import annotations

import logging
from typing import Any

from iris_reply.core.llm_client import LLMClient
from iris_reply.core.memory_api import MemoryAPI
from iris_reply.core.message_builder import MessageBuilder
from iris_reply.models.models import (
    AssembledContext,
    FollowupPlan,
    FollowupType,
)
from iris_reply.config.config import ReplyConfig

logger = logging.getLogger("iris_reply.followup_planner")


class FollowupPlanner:
    def __init__(
        self,
        llm_client: LLMClient,
        message_builder: MessageBuilder,
        memory_api: MemoryAPI,
        config: ReplyConfig,
    ):
        self._llm = llm_client
        self._builder = message_builder
        self._memory = memory_api
        self._config = config

    async def plan(
        self,
        group_id: str,
        bot_reply: str,
    ) -> FollowupPlan | None:
        context = await self._assemble_context(group_id, bot_reply)
        system_prompt, prompt = self._builder.build_followup_plan_prompt(
            bot_reply, context
        )
        provider_id = self._config.followup.get("llm_provider_id", "")
        result = await self._llm.generate_json(
            prompt, system_prompt, provider_id, module="followup_plan"
        )

        if result is None:
            return None

        should_followup = result.get("should_followup", False)
        if not should_followup:
            return FollowupPlan(should_followup=False)

        try:
            ft = FollowupType(result.get("followup_type", "topic_extend"))
        except ValueError:
            ft = FollowupType.TOPIC_EXTEND

        delay_min = self._config.followup.get("delay_seconds_min", 30)
        delay_max = self._config.followup.get("delay_seconds_max", 120)
        delay = int(result.get("delay_seconds", 60))
        delay = max(delay_min, min(delay_max, delay))

        max_wait = int(result.get("max_wait_messages", 5))
        max_wait = max(3, min(8, max_wait))

        return FollowupPlan(
            should_followup=True,
            followup_type=ft,
            delay_seconds=delay,
            direction=result.get("direction", ""),
            max_wait_messages=max_wait,
        )

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
