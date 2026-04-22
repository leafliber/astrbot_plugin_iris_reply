from __future__ import annotations

import logging
from typing import Any

from iris_reply.core.memory_api import MemoryAPI
from iris_reply.models.models import AssembledContext
from iris_reply.config.config import ReplyConfig

logger = logging.getLogger("iris_reply.context_assembler")


class ContextAssembler:
    def __init__(self, memory_api: MemoryAPI, config: ReplyConfig):
        self._memory = memory_api
        self._config = config

    async def assemble(
        self,
        group_id: str,
        user_id: str,
        query: str,
    ) -> AssembledContext:
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
