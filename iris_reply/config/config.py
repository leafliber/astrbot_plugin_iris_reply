from __future__ import annotations

import copy
import logging
from typing import Any

from astrbot.api.star import Context

_DEFAULTS: dict[str, Any] = {
    "proactive": {
        "enable": False,
        "mode": "rule",
        "min_interval_seconds": 300,
        "max_replies_per_hour": 3,
        "group_whitelist_mode": True,
        "llm_provider_id": "",
        "analysis_window_messages": 10,
        "relevance_threshold": 0.6,
        "intent_threshold": 0.5,
        "silence_gap_seconds": 120,
    },
    "followup": {
        "enable": False,
        "delay_seconds_min": 30,
        "delay_seconds_max": 120,
        "max_wait_messages": 5,
        "followup_after_all_replies": False,
        "llm_provider_id": "",
    },
    "keyword": {
        "enable": True,
        "static_keywords": ["帮我", "怎么办", "求助", "请问"],
        "dynamic_generation": True,
        "dynamic_refresh_interval_minutes": 60,
        "validation_mode": "llm",
        "validation_threshold": 0.7,
        "llm_provider_id": "",
        "max_keywords_per_group": 50,
    },
    "memory": {
        "integration_mode": "auto",
        "fallback_on_unavailable": True,
        "max_context_messages": 10,
        "max_memory_results": 5,
        "max_knowledge_facts": 3,
    },
    "cooldown": {
        "default_seconds": 300,
    },
}

logger = logging.getLogger("iris_reply.config")


class ReplyConfig:
    def __init__(self, context: Context):
        self._context = context
        self._raw: dict[str, Any] = copy.deepcopy(_DEFAULTS)
        self._load()

    def _load(self) -> None:
        try:
            user_config = self._context.get_config()
            if user_config and isinstance(user_config, dict):
                self._merge(self._raw, user_config)
        except Exception as e:
            logger.warning("加载用户配置失败，使用默认值: %s", e)

    def _merge(self, base: dict, override: dict) -> dict:
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._merge(base[k], v)
            else:
                base[k] = v
        return base

    @property
    def proactive(self) -> dict[str, Any]:
        return self._raw["proactive"]

    @property
    def followup(self) -> dict[str, Any]:
        return self._raw["followup"]

    @property
    def keyword(self) -> dict[str, Any]:
        return self._raw["keyword"]

    @property
    def memory(self) -> dict[str, Any]:
        return self._raw["memory"]

    @property
    def cooldown(self) -> dict[str, Any]:
        return self._raw["cooldown"]

    def reload(self) -> None:
        self._raw = copy.deepcopy(_DEFAULTS)
        self._load()
