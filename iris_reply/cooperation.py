from __future__ import annotations

from typing import Any, Dict, List, Optional

from astrbot.api import logger
from astrbot.api.star import Context

from .models import FollowUpDecision, FollowUpExpectation, FollowUpReplyType


class MemoryCooperation:
    def __init__(self, context: Context, config: Dict[str, Any]) -> None:
        self._context = context
        self._config = config
        self._available: Optional[bool] = None
        self._component_manager = None
        self._l1_buffer = None
        self._l2_adapter = None
        self._profile_storage = None

    def _check_available(self) -> bool:
        if self._available is not None:
            return self._available

        try:
            from iris_memory.core.lifecycle import get_component_manager

            cm = get_component_manager()
            self._component_manager = cm
            self._available = True
            logger.info("iris_chat_memory detected, cooperation enabled")
            return True
        except Exception as e:
            logger.debug(f"iris_chat_memory not available: {e}")
            self._available = False
            return False

    def _get_l1_buffer(self):
        if self._l1_buffer is not None:
            return self._l1_buffer if self._l1_buffer.is_available else None

        if self._component_manager is None:
            return None

        try:
            comp = self._component_manager.get_component("l1")
            if comp and comp.is_available:
                self._l1_buffer = comp
                return comp
        except Exception as e:
            logger.debug(f"Failed to get L1 buffer component: {e}")

        self._l1_buffer = None
        return None

    def _get_l2_adapter(self):
        if self._l2_adapter is not None:
            return self._l2_adapter if self._l2_adapter.is_available else None

        if self._component_manager is None:
            return None

        try:
            comp = self._component_manager.get_component("l2")
            if comp and comp.is_available:
                self._l2_adapter = comp
                return comp
        except Exception as e:
            logger.debug(f"Failed to get L2 adapter component: {e}")

        self._l2_adapter = None
        return None

    def _get_profile_storage(self):
        if self._profile_storage is not None:
            return self._profile_storage if self._profile_storage.is_available else None

        if self._component_manager is None:
            return None

        try:
            comp = self._component_manager.get_component("profile")
            if comp and comp.is_available:
                self._profile_storage = comp
                return comp
        except Exception as e:
            logger.debug(f"Failed to get profile storage component: {e}")

        self._profile_storage = None
        return None

    async def get_recent_messages(
        self, group_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        if not self._config.get("cooperation_context_enabled", True):
            return []

        if not self._check_available():
            return []

        l1 = self._get_l1_buffer()
        if l1 is None:
            return []

        try:
            messages = l1.get_context(group_id, max_length=limit)
            if not messages:
                return []

            result = []
            for msg in messages:
                item: Dict[str, Any] = {
                    "role": msg.role,
                    "content": msg.content,
                    "source": msg.source,
                }
                if msg.metadata:
                    sender_name = msg.metadata.get("nickname") or msg.metadata.get("user_name", "")
                    if sender_name:
                        item["sender_name"] = sender_name
                    sender_id = msg.metadata.get("user_id", "")
                    if sender_id:
                        item["sender_id"] = sender_id
                result.append(item)

            return result
        except Exception as e:
            logger.debug(f"Failed to get recent messages from iris_chat_memory: {e}")
            return []

    async def get_user_profile(
        self, user_id: str, group_id: str = "default"
    ) -> Optional[Dict[str, Any]]:
        if not self._config.get("cooperation_profile_enabled", True):
            return None

        if not self._check_available():
            return None

        storage = self._get_profile_storage()
        if storage is None:
            return None

        try:
            from iris_memory.profile.user_profile import UserProfileManager

            manager = UserProfileManager(storage)
            profile = await manager.get_or_create(user_id, group_id)

            result: Dict[str, Any] = {"user_id": profile.user_id}
            if profile.user_name:
                result["user_name"] = profile.user_name
            if profile.personality_tags:
                result["personality_tags"] = profile.personality_tags
            if profile.interests:
                result["interests"] = profile.interests
            if profile.occupation:
                result["occupation"] = profile.occupation
            if profile.language_style:
                result["language_style"] = profile.language_style
            if profile.bot_relationship:
                result["bot_relationship"] = profile.bot_relationship

            return result
        except Exception as e:
            logger.debug(f"Failed to get user profile from iris_chat_memory: {e}")
            return None

    async def get_group_profile(
        self, group_id: str
    ) -> Optional[Dict[str, Any]]:
        if not self._config.get("cooperation_profile_enabled", True):
            return None

        if not self._check_available():
            return None

        storage = self._get_profile_storage()
        if storage is None:
            return None

        try:
            from iris_memory.profile.group_profile import GroupProfileManager

            manager = GroupProfileManager(storage)
            profile = await manager.get_or_create(group_id)

            result: Dict[str, Any] = {"group_id": profile.group_id}
            if profile.group_name:
                result["group_name"] = profile.group_name
            if profile.interests:
                result["interests"] = profile.interests
            if profile.atmosphere_tags:
                result["atmosphere_tags"] = profile.atmosphere_tags
            if profile.long_term_tags:
                result["long_term_tags"] = profile.long_term_tags
            if profile.blacklist_topics:
                result["blacklist_topics"] = profile.blacklist_topics

            return result
        except Exception as e:
            logger.debug(f"Failed to get group profile from iris_chat_memory: {e}")
            return None

    async def search_memories(
        self, query: str, group_id: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if not self._config.get("cooperation_context_enabled", True):
            return []

        if not self._check_available():
            return []

        l2 = self._get_l2_adapter()
        if l2 is None:
            return []

        try:
            results = await l2.retrieve(query, group_id=group_id, top_k=top_k)
            if not results:
                return []

            return [
                {
                    "content": r.entry.content,
                    "score": r.score,
                    "metadata": r.entry.metadata,
                }
                for r in results
            ]
        except Exception as e:
            logger.debug(f"Failed to search memories from iris_chat_memory: {e}")
            return []

    async def llm_confirm_proactive(
        self,
        group_id: str,
        signals: List[Any],
        recent_messages: List[Dict[str, Any]],
    ) -> bool:
        if not self._config.get("cooperation_llm_confirm_enabled", True):
            return False

        try:
            prompt = self._build_llm_confirm_prompt(signals, recent_messages)
            result = await self._call_llm(prompt)
            if result is None:
                return False
            return self._parse_confirm_response(result)
        except Exception as e:
            logger.warning(f"LLM confirm failed: {e}")
            return False

    async def llm_decide_followup(
        self, expectation: FollowUpExpectation
    ) -> Optional[FollowUpDecision]:
        if not self._config.get("cooperation_llm_followup_enabled", True):
            return None

        try:
            from .followup import build_followup_prompt, parse_followup_response

            prompt = build_followup_prompt(expectation)
            result = await self._call_llm(prompt)
            if result is None:
                return None
            return parse_followup_response(result)
        except Exception as e:
            logger.warning(f"LLM followup decide failed: {e}")
            return None

    async def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            provider = None
            provider_id = self._config.get("provider_id", "")
            if provider_id:
                try:
                    provider = self._context.get_using_provider(provider_id)
                except Exception:
                    provider = None

            if provider is None:
                provider = self._context.get_using_provider()

            if provider is None:
                return None

            response = await provider.text_chat(
                prompt=prompt,
                session_id="proactive_reply_system",
            )

            if hasattr(response, "completion_text"):
                return response.completion_text
            if isinstance(response, str):
                return response
            return str(response) if response else None
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    @staticmethod
    def _build_llm_confirm_prompt(
        signals: List[Any],
        recent_messages: List[Dict[str, Any]],
    ) -> str:
        signal_desc = "\n".join(
            f"  - {getattr(s, 'signal_type', 'unknown')}: "
            f"weight={getattr(s, 'weight', 0):.2f}"
            for s in signals
        )

        messages_desc = "\n".join(
            f"  {m.get('sender_id', '用户')}: {m.get('content', '')}"
            for m in recent_messages[-5:]
        )

        return (
            "你是一个在群聊中具有独立人格的 AI 助手，需要判断是否应该主动回复群聊消息。\n\n"
            f"【检测到的信号】\n{signal_desc}\n\n"
            f"【近期群聊消息】\n{messages_desc}\n\n"
            "请判断你是否应该主动回复。输出 JSON：\n"
            '{"should_reply": true/false, "reason": "简短原因"}\n\n'
            "判断标准：\n"
            "- 只在确实有意义的场景下回复，比如有人提问、表达情感、寻求注意\n"
            "- 不要过度主动，避免在正常聊天中频繁插话\n"
            "- 如果话题已经结束或用户不需要帮助，不要回复\n"
            "- 如果用户只是在和其他人闲聊且没有涉及你，不要强行介入"
        )

    @staticmethod
    def _parse_confirm_response(content: str) -> bool:
        try:
            import json

            text = content.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start < 0 or end <= start:
                return False

            data = json.loads(text[start:end])
            return bool(data.get("should_reply", False))
        except Exception:
            return False
