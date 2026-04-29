from __future__ import annotations

from typing import Callable, Dict, Optional

from astrbot.api import logger
from astrbot.api.event import MessageChain
from astrbot.api.star import Context

from .models import ProactiveReplyResult


class ReplySender:
    def __init__(self, context: Context) -> None:
        self._context = context
        self._umo_map: Dict[str, str] = {}

    def store_umo(self, group_id: str, unified_msg_origin: str) -> None:
        self._umo_map[group_id] = unified_msg_origin

    def get_umo(self, group_id: str) -> Optional[str]:
        return self._umo_map.get(group_id)

    async def send_proactive_reply(
        self,
        result: ProactiveReplyResult,
        llm_call: Optional[Callable] = None,
    ) -> bool:
        try:
            umo = self._umo_map.get(result.group_id)
            if not umo:
                logger.warning(
                    f"No unified_msg_origin stored for group {result.group_id}, "
                    f"cannot send proactive reply"
                )
                return False

            reply_text = await self._generate_reply_text(result, llm_call)
            if not reply_text:
                logger.warning("LLM generated empty reply, skipping")
                return False

            if reply_text.upper() == "PASS":
                logger.info("LLM decided to skip this reply (PASS)")
                return False

            chain = MessageChain().message(reply_text)
            await self._context.send_message(umo, chain)

            logger.info(
                f"Proactive reply sent: group={result.group_id}, "
                f"source={result.source}, length={len(reply_text)}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to send proactive reply: {e}")
            return False

    async def _generate_reply_text(
        self,
        result: ProactiveReplyResult,
        llm_call: Optional[Callable] = None,
    ) -> Optional[str]:
        if llm_call is not None:
            try:
                response = await llm_call(
                    prompt=result.trigger_prompt,
                    **result.reply_params,
                )
                if isinstance(response, str):
                    return response.strip()
                if hasattr(response, "text"):
                    return response.text.strip()
                if hasattr(response, "content"):
                    return str(response.content).strip()
                return str(response).strip()
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return None

        return None
