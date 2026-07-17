from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from astrbot.api import logger


@dataclass
class TriggerResult:
    """LLM 触发评估的结构化结果。"""

    should_reply: bool = False
    observation: str = ""
    follow_up_users: list[str] = field(default_factory=list)
    follow_up_keywords: list[str] = field(default_factory=list)
    interest_reason: str = ""
    topic_drifted: bool = False
    parse_failed: bool = False


def extract_json(text: str) -> dict | None:
    """从 LLM 输出中提取第一个合法 JSON 对象。

    依次尝试：markdown 代码块剥离 -> True/False 修正 -> 整体解析 ->
    括号配平扫描（跳过字符串内部的花括号）。
    """
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    text = re.sub(r'"\s*:\s*True', '": true', text)
    text = re.sub(r'"\s*:\s*False', '": false', text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    in_string = False
    escape = False
    brace_count = 0
    start = -1
    for i, c in enumerate(text):
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            if brace_count == 0:
                start = i
            brace_count += 1
        elif c == "}":
            brace_count -= 1
            if brace_count == 0 and start >= 0:
                try:
                    obj = json.loads(text[start : i + 1])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    start = -1

    return None


def parse_bool(val: Any) -> bool:
    if isinstance(val, str):
        return val.lower() in ("true", "yes", "1")
    if isinstance(val, (int, float)):
        return bool(val)
    return bool(val)


def parse_string_list(raw: Any, max_len: int = 10) -> list[str]:
    if not isinstance(raw, list):
        return []
    result = [str(u).strip() for u in raw if str(u).strip()]
    return result[:max_len]


def parse_trigger(text: str) -> TriggerResult:
    """解析 LLM 触发评估输出，兼容多种字段命名。"""
    obj = extract_json(text)
    if not obj:
        logger.warning("Iris Reply: trigger JSON parse failed, raw text: %.300s", text)
        return TriggerResult(parse_failed=True)

    should_reply = parse_bool(
        obj.get("reply", obj.get("should_reply", False))
    )
    topic_drifted = parse_bool(
        obj.get("drifted", obj.get("topic_drifted", False))
    )
    follow_up_users = parse_string_list(
        obj.get("watch", obj.get("follow_up_users", []))
    )
    follow_up_keywords = parse_string_list(
        obj.get("watch_keywords", obj.get("follow_up_keywords", [])),
        max_len=10,
    )
    interest_reason = str(
        obj.get("why", obj.get("interest_reason", ""))
    )
    observation = str(obj.get("obs", obj.get("observation", "")))

    return TriggerResult(
        should_reply=should_reply,
        observation=observation,
        follow_up_users=follow_up_users,
        follow_up_keywords=follow_up_keywords,
        interest_reason=interest_reason,
        topic_drifted=topic_drifted,
    )
