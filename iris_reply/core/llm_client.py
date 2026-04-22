from __future__ import annotations

import json
import logging
from typing import Any

from astrbot.api.star import Context

logger = logging.getLogger("iris_reply.llm_client")


class LLMClient:
    def __init__(self, context: Context):
        self._context = context

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        provider_id: str = "",
        module: str = "reply",
    ) -> str:
        try:
            pid = await self._resolve_provider_id(provider_id)
            llm_resp = await self._context.llm_generate(
                chat_provider_id=pid,
                prompt=prompt,
                system_prompt=system_prompt,
            )
            if llm_resp and llm_resp.completion_text:
                return llm_resp.completion_text.strip()
            return ""
        except Exception as e:
            logger.error("LLM 调用失败 [%s]: %s", module, e)
            return ""

    async def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        provider_id: str = "",
        module: str = "reply",
    ) -> dict[str, Any] | None:
        raw = await self.generate(prompt, system_prompt, provider_id, module)
        if not raw:
            return None
        try:
            cleaned = _extract_json(raw)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("LLM 返回非有效 JSON: %s", raw[:200])
            return None

    async def _resolve_provider_id(self, provider_id: str = "") -> str:
        if provider_id:
            return provider_id
        try:
            providers = self._context.get_all_providers()
            if providers:
                first = providers[0]
                meta = first.meta()
                if meta and meta.id:
                    return str(meta.id)
        except Exception:
            pass
        return ""


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text
