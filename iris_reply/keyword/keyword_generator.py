from __future__ import annotations

import logging
from typing import Any

from iris_reply.core.llm_client import LLMClient
from iris_reply.core.message_builder import MessageBuilder
from iris_reply.core.memory_api import MemoryAPI
from iris_reply.models.models import AssembledContext, KeywordSource
from iris_reply.config.config import ReplyConfig

logger = logging.getLogger("iris_reply.keyword_generator")


class KeywordGenerator:
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

    async def generate_from_profile(self, group_id: str) -> list[tuple[str, KeywordSource, float]]:
        profile = await self._memory.get_group_profile(group_id)
        if not profile:
            return []
        keywords = []
        interests = profile.get("interests", [])
        if isinstance(interests, list):
            for interest in interests[:5]:
                if isinstance(interest, str) and interest.strip():
                    keywords.append((interest.strip(), KeywordSource.PROFILE, 0.7))
        traits = profile.get("traits", [])
        if isinstance(traits, list):
            for trait in traits[:3]:
                if isinstance(trait, str) and trait.strip():
                    keywords.append((trait.strip(), KeywordSource.PROFILE, 0.6))
        return keywords

    async def generate_from_conversation(self, group_id: str, existing_keywords: list[str]) -> list[tuple[str, KeywordSource, float]]:
        max_ctx = self._config.memory.get("max_context_messages", 10)
        recent = await self._memory.get_recent_context(group_id, max_ctx)
        if not recent:
            return []

        context = AssembledContext(recent_messages=recent)
        system_prompt, prompt = self._builder.build_keyword_generation_prompt(
            context, existing_keywords
        )
        provider_id = self._config.keyword.get("llm_provider_id", "")
        result = await self._llm.generate_json(
            prompt, system_prompt, provider_id, module="keyword_generate"
        )

        if result is None or "keywords" not in result:
            return []

        keywords = []
        for kw in result["keywords"][:10]:
            if isinstance(kw, str) and kw.strip() and kw not in existing_keywords:
                keywords.append((kw.strip(), KeywordSource.CONVERSATION, 0.5))
        return keywords

    async def generate_from_knowledge(self, group_id: str) -> list[tuple[str, KeywordSource, float]]:
        facts = await self._memory.query_knowledge_graph("", group_id)
        if not facts:
            return []
        keywords = []
        for fact in facts[:5]:
            subject = fact.get("subject", "")
            if subject and subject.strip():
                keywords.append((subject.strip(), KeywordSource.KNOWLEDGE, 0.6))
        return keywords

    async def refresh_all(self, group_id: str, existing_keywords: list[str]) -> list[tuple[str, KeywordSource, float]]:
        all_keywords: list[tuple[str, KeywordSource, float]] = []
        seen: set[str] = set(existing_keywords)

        profile_kws = await self.generate_from_profile(group_id)
        for kw in profile_kws:
            if kw[0] not in seen:
                seen.add(kw[0])
                all_keywords.append(kw)

        conv_kws = await self.generate_from_conversation(group_id, list(seen))
        for kw in conv_kws:
            if kw[0] not in seen:
                seen.add(kw[0])
                all_keywords.append(kw)

        knowledge_kws = await self.generate_from_knowledge(group_id)
        for kw in knowledge_kws:
            if kw[0] not in seen:
                seen.add(kw[0])
                all_keywords.append(kw)

        max_kw = self._config.keyword.get("max_keywords_per_group", 50)
        if len(all_keywords) > max_kw:
            all_keywords.sort(key=lambda x: x[2], reverse=True)
            all_keywords = all_keywords[:max_kw]

        logger.info("群 %s 动态关键词刷新完成，共 %d 个", group_id, len(all_keywords))
        return all_keywords
