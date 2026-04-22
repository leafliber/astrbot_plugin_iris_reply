from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger("iris_reply.memory_api")


class MemoryAPI:
    def __init__(self):
        self._conversation_manager: Any | None = None
        self._available: bool = False

    def bind(self, conversation_manager: Any | None) -> None:
        self._conversation_manager = conversation_manager
        self._available = conversation_manager is not None
        if self._available:
            logger.info("Memory API 已绑定 tier_memory ConversationManager")
        else:
            logger.info("Memory API 未绑定，将降级运行")

    @property
    def is_available(self) -> bool:
        return self._available and self._conversation_manager is not None

    async def get_recent_context(self, group_id: str, limit: int = 10) -> list[dict[str, Any]]:
        if not self.is_available:
            return []
        try:
            cm = self._conversation_manager
            l1 = cm.l1_buffer
            if l1 is None:
                return []
            messages = await l1.get_recent_messages(group_id, limit)
            return [_format_l1_message(m) for m in messages]
        except Exception as e:
            logger.debug("获取 L1 上下文失败: %s", e)
            return []

    async def search_memory(self, query: str, group_id: str, limit: int = 5) -> list[dict[str, Any]]:
        if not self.is_available:
            return []
        try:
            cm = self._conversation_manager
            l2 = cm.l2_memory
            if l2 is None:
                return []
            results = await l2.search(query, group_id, limit)
            return [_format_l2_result(r) for r in results]
        except Exception as e:
            logger.debug("搜索 L2 记忆失败: %s", e)
            return []

    async def query_knowledge_graph(self, entity: str, group_id: str) -> list[dict[str, Any]]:
        if not self.is_available:
            return []
        try:
            cm = self._conversation_manager
            l3 = cm.l3_kg
            if l3 is None:
                return []
            results = await l3.query(entity, group_id)
            return [_format_l3_result(r) for r in results]
        except Exception as e:
            logger.debug("查询 L3 知识图谱失败: %s", e)
            return []

    async def get_user_profile(self, user_id: str, group_id: str) -> dict[str, Any] | None:
        if not self.is_available:
            return None
        try:
            cm = self._conversation_manager
            profile_mgr = cm.profile_manager
            if profile_mgr is None:
                return None
            profile = await profile_mgr.get_user_profile(user_id, group_id)
            if profile is None:
                return None
            return _format_profile(profile)
        except Exception as e:
            logger.debug("获取用户画像失败: %s", e)
            return None

    async def get_group_profile(self, group_id: str) -> dict[str, Any] | None:
        if not self.is_available:
            return None
        try:
            cm = self._conversation_manager
            profile_mgr = cm.profile_manager
            if profile_mgr is None:
                return None
            profile = await profile_mgr.get_group_profile(group_id)
            if profile is None:
                return None
            return _format_profile(profile)
        except Exception as e:
            logger.debug("获取群画像失败: %s", e)
            return None

    async def notify_bot_message(self, group_id: str, content: str, role: str = "assistant") -> None:
        if not self.is_available:
            return
        try:
            cm = self._conversation_manager
            l1 = cm.l1_buffer
            if l1 is not None:
                await l1.add_message(group_id, role, content)
        except Exception as e:
            logger.debug("通知 tier_memory Bot 消息失败: %s", e)


def _format_l1_message(msg: Any) -> dict[str, Any]:
    if isinstance(msg, dict):
        return msg
    result: dict[str, Any] = {}
    for attr in ("role", "content", "sender_id", "sender_name", "timestamp"):
        val = getattr(msg, attr, None)
        if val is not None:
            result[attr] = val
    return result


def _format_l2_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    out: dict[str, Any] = {}
    for attr in ("content", "summary", "relevance", "timestamp", "metadata"):
        val = getattr(result, attr, None)
        if val is not None:
            out[attr] = val
    return out


def _format_l3_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    out: dict[str, Any] = {}
    for attr in ("subject", "predicate", "object", "fact", "relevance"):
        val = getattr(result, attr, None)
        if val is not None:
            out[attr] = val
    return out


def _format_profile(profile: Any) -> dict[str, Any]:
    if isinstance(profile, dict):
        return profile
    out: dict[str, Any] = {}
    for attr in ("user_id", "group_id", "traits", "interests", "summary", "metadata"):
        val = getattr(profile, attr, None)
        if val is not None:
            out[attr] = val
    if hasattr(profile, "to_dict") and callable(profile.to_dict):
        return profile.to_dict()
    return out
