from __future__ import annotations

import logging
from typing import Any

from iris_reply.core.llm_client import LLMClient
from iris_reply.core.memory_api import MemoryAPI
from iris_reply.core.message_builder import MessageBuilder
from iris_reply.models.models import (
    AnalysisResult,
    AssembledContext,
    Decision,
)
from iris_reply.config.config import ReplyConfig

logger = logging.getLogger("iris_reply.decision_engine")


class DecisionEngine:
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

    async def decide(
        self,
        analysis: AnalysisResult,
        group_id: str,
        user_id: str,
    ) -> Decision:
        if not analysis.should_consider:
            return Decision(should_reply=False, reason="分析信号不足")

        context = await self._assemble_context(group_id, user_id, analysis.summary)
        system_prompt, prompt = self._builder.build_proactive_decision_prompt(
            analysis, context
        )
        provider_id = self._config.proactive.get("llm_provider_id", "")
        result = await self._llm.generate_json(
            prompt, system_prompt, provider_id, module="proactive_decide"
        )

        if result is None:
            return Decision(
                should_reply=False,
                reason="LLM 决策返回无效",
            )

        should_reply = result.get("should_reply", False)
        confidence = float(result.get("confidence", 0.0))

        return Decision(
            should_reply=should_reply,
            confidence=confidence,
            reason=result.get("reason", ""),
            reply_direction=result.get("reply_direction"),
        )

    async def generate_reply(
        self,
        analysis: AnalysisResult,
        group_id: str,
        user_id: str,
        direction: str,
    ) -> str:
        context = await self._assemble_context(group_id, user_id, analysis.summary)
        system_prompt, prompt = self._builder.build_proactive_reply_prompt(
            analysis, context, direction
        )
        provider_id = self._config.proactive.get("llm_provider_id", "")
        reply = await self._llm.generate(
            prompt, system_prompt, provider_id, module="proactive_reply"
        )
        return reply

    async def _assemble_context(self, group_id: str, user_id: str, query: str) -> AssembledContext:
        max_ctx = self._config.memory.get("max_context_messages", 10)
        max_mem = self._config.memory.get("max_memory_results", 5)
        max_kg = self._config.memory.get("max_knowledge_facts", 3)

        recent = await self._memory.get_recent_context(group_id, max_ctx)
        memories = await self._memory.search_memory(query, group_id, max_mem)
        facts = await self._memory.query_knowledge_graph(query, group_id)
        facts = facts[:max_kg]
        user_profile = await self._memory.get_user_profile(user_id, group_id)
        group_profile = await self._memory.get_group_profile(group_id)

        return AssembledContext(
            recent_messages=recent,
            relevant_memories=memories,
            knowledge_facts=facts,
            user_profile=user_profile,
            group_profile=group_profile,
        )
