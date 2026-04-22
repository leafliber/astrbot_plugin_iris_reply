from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from astrbot.api.star import Context
from astrbot.core.message.components import Plain
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.platform.message_type import MessageType

if TYPE_CHECKING:
    from iris_reply.core.memory_api import MemoryAPI

logger = logging.getLogger("iris_reply.reply_sender")


class ReplySender:
    def __init__(self, context: Context, memory_api: MemoryAPI):
        self._context = context
        self._memory_api = memory_api
        self._session_cache: dict[str, str] = {}

    def register_session(self, group_id: str, unified_msg_origin: str) -> None:
        self._session_cache[group_id] = unified_msg_origin

    async def send_group_message(self, group_id: str, content: str) -> bool:
        if not content or not content.strip():
            logger.warning("尝试发送空消息，跳过")
            return False
        try:
            await self._do_send(group_id, content)
            await self._memory_api.notify_bot_message(group_id, content, "assistant")
            logger.info("已发送消息到群 %s: %s", group_id, content[:50])
            return True
        except Exception as e:
            logger.error("发送群消息失败 [%s]: %s", group_id, e)
            return False

    async def _do_send(self, group_id: str, content: str) -> None:
        umo = self._session_cache.get(group_id)
        if umo:
            chain = MessageChain(chain=[Plain(content)])
            await self._context.send_message(umo, chain)
            return

        for platform in self._context.platform_manager.platform_insts:
            session = MessageSession(
                platform_name=platform.meta().id,
                message_type=MessageType.GROUP_MESSAGE,
                session_id=group_id,
            )
            try:
                chain = MessageChain(chain=[Plain(content)])
                await self._context.send_message(str(session), chain)
                self._session_cache[group_id] = str(session)
                return
            except Exception as e:
                logger.debug("尝试通过平台 %s 发送失败: %s", platform.meta().id, e)
                continue

        logger.error("找不到群 %s 对应的平台或 session", group_id)
