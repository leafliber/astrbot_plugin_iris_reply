from __future__ import annotations

import logging
from typing import Any

from iris_reply.core.llm_client import LLMClient
from iris_reply.core.message_builder import MessageBuilder
from iris_reply.core.memory_api import MemoryAPI
from iris_reply.models.models import (
    AssembledContext,
    KeywordMatch,
    ValidationResult,
)
from iris_reply.config.config import ReplyConfig

logger = logging.getLogger("iris_reply.keyword_validator")


class KeywordValidator:
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

    async def validate(
        self,
        match: KeywordMatch,
        message: str,
        group_id: str,
        user_id: str,
    ) -> ValidationResult:
        force_keywords = self._config.keyword.get("force_reply_keywords", [])
        if force_keywords:
            message_lower = message.lower()
            for fk in force_keywords:
                if fk.lower() in message_lower:
                    logger.info("命中强制回复关键词: %s", fk)
                    return ValidationResult(
                        should_reply=True,
                        confidence=1.0,
                        reason=f"命中强制回复关键词 '{fk}'",
                        reply_direction=f"用户消息包含强制关键词 '{fk}'",
                    )

        mode = self._config.keyword.get("validation_mode", "llm")

        if mode == "rule":
            return self._rule_validate(match, message)
        return await self._llm_validate(match, message, group_id, user_id)

    def _rule_validate(self, match: KeywordMatch, message: str) -> ValidationResult:
        if match.match_type.value == "exact" and match.confidence >= 0.9:
            return ValidationResult(
                should_reply=True,
                confidence=0.8,
                reason=f"精确匹配关键词 '{match.keyword}'",
                reply_direction=f"用户消息中包含关键词 '{match.keyword}'",
            )
        if match.confidence >= 0.7:
            return ValidationResult(
                should_reply=True,
                confidence=match.confidence * 0.8,
                reason=f"模糊匹配关键词 '{match.keyword}'",
                reply_direction=f"用户消息可能涉及 '{match.keyword}'",
            )
        return ValidationResult(
            should_reply=False,
            confidence=match.confidence,
            reason=f"关键词 '{match.keyword}' 匹配置信度不足",
        )

    async def _llm_validate(
        self,
        match: KeywordMatch,
        message: str,
        group_id: str,
        user_id: str,
    ) -> ValidationResult:
        context = await self._assemble_context(group_id, user_id, match.keyword)
        system_prompt, prompt = self._builder.build_keyword_validation_prompt(
            match, message, context
        )
        provider_id = self._config.keyword.get("llm_provider_id", "")
        result = await self._llm.generate_json(
            prompt, system_prompt, provider_id, module="keyword_validate"
        )

        if result is None:
            logger.warning("关键词验证 LLM 返回无效，降级为规则验证")
            return self._rule_validate(match, message)

        threshold = self._config.keyword.get("validation_threshold", 0.7)
        should_reply = result.get("should_reply", False)
        confidence = float(result.get("confidence", 0.0))

        if should_reply and confidence < threshold:
            should_reply = False

        return ValidationResult(
            should_reply=should_reply,
            confidence=confidence,
            reason=result.get("reason", ""),
            reply_direction=result.get("reply_direction"),
        )

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
