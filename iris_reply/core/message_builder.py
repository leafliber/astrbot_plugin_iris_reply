from __future__ import annotations

import json
from typing import Any

from iris_reply.models.models import (
    AnalysisResult,
    AssembledContext,
    KeywordMatch,
    FollowupPlan,
)


class MessageBuilder:

    # ── 主动接入决策 Prompt ──

    def build_proactive_decision_prompt(
        self,
        analysis: AnalysisResult,
        context: AssembledContext,
        bot_persona: str = "",
    ) -> tuple[str, str]:
        system_prompt = (
            "你是一个群聊参与决策助手。根据以下信息判断 Bot 是否应该主动参与对话。\n"
            "你必须只输出 JSON，不要输出任何其他内容。"
        )
        recent = _format_messages(context.recent_messages)
        memories = _format_memories(context.relevant_memories)
        group_info = _format_profile(context.group_profile, "群聊")
        user_info = _format_profile(context.user_profile, "用户")

        prompt = f"""## Bot 人设
{bot_persona or '一个友好的群聊助手'}

## 近期对话
{recent or '（无）'}

## 相关记忆
{memories or '（无）'}

## 群聊画像
{group_info or '（无）'}

## 用户画像
{user_info or '（无）'}

## 分析信号
- 话题相关性: {analysis.topic_relevance:.2f}
- 意图信号: {analysis.intent_signal:.2f}
- 情感信号: {analysis.emotion_signal:.2f}
- 沉默间隔: {analysis.silence_gap:.0f}秒
- 提及信号: {analysis.mention_signal:.2f}

## 决策要求
请判断 Bot 是否应该主动回复。输出 JSON：
{{
    "should_reply": true或false,
    "confidence": 0.0到1.0,
    "reason": "决策原因",
    "reply_direction": "回复方向提示（如果 should_reply 为 true）"
}}"""
        return system_prompt, prompt

    # ── 主动回复生成 Prompt ──

    def build_proactive_reply_prompt(
        self,
        analysis: AnalysisResult,
        context: AssembledContext,
        direction: str,
        bot_persona: str = "",
    ) -> tuple[str, str]:
        system_prompt = (
            "你是一个群聊助手，正在主动参与群聊对话。请根据上下文自然地回复。\n"
            "回复要求：\n"
            "1. 语气自然，像群聊参与者而非机器人\n"
            "2. 不要过于正式或冗长\n"
            "3. 回复应与当前话题相关\n"
            "4. 不要提及你是被触发回复的"
        )
        recent = _format_messages(context.recent_messages)
        memories = _format_memories(context.relevant_memories)
        group_info = _format_profile(context.group_profile, "群聊")
        user_info = _format_profile(context.user_profile, "用户")

        prompt = f"""## Bot 人设
{bot_persona or '一个友好的群聊助手'}

## 近期对话
{recent or '（无）'}

## 相关记忆
{memories or '（无）'}

## 群聊画像
{group_info or '（无）'}

## 用户画像
{user_info or '（无）'}

## 回复方向
{direction}

## 消息摘要
{analysis.summary or '（无）'}

请直接输出你的回复内容，不要包含任何前缀或标记。"""
        return system_prompt, prompt

    # ── 关键词验证 Prompt ──

    def build_keyword_validation_prompt(
        self,
        match: KeywordMatch,
        message: str,
        context: AssembledContext,
    ) -> tuple[str, str]:
        system_prompt = (
            "你是一个关键词响应验证助手。消息中命中了关键词，请判断是否需要 Bot 回复。\n"
            "你必须只输出 JSON，不要输出任何其他内容。"
        )
        recent = _format_messages(context.recent_messages)
        user_info = _format_profile(context.user_profile, "用户")

        prompt = f"""## 命中关键词
"{match.keyword}"（匹配方式: {match.match_type.value}，来源: {match.source.value}）

## 消息内容
{message}

## 上下文
{recent or '（无）'}

## 用户画像
{user_info or '（无）'}

## 验证要求
1. 关键词是否在上下文中被自然使用（非偶然出现）
2. 用户是否有隐式求助意图
3. Bot 回复是否能提供价值
4. 回复是否自然、不突兀

输出 JSON：
{{
    "should_reply": true或false,
    "confidence": 0.0到1.0,
    "reason": "验证原因",
    "reply_direction": "回复方向提示"
}}"""
        return system_prompt, prompt

    # ── 关键词回复生成 Prompt ──

    def build_keyword_reply_prompt(
        self,
        match: KeywordMatch,
        message: str,
        context: AssembledContext,
        direction: str,
        bot_persona: str = "",
    ) -> tuple[str, str]:
        system_prompt = (
            "你是一个群聊助手，检测到消息中有关键词需要回复。请根据上下文自然地回复。\n"
            "回复要求：\n"
            "1. 语气自然，像群聊参与者\n"
            "2. 不要过于正式或冗长\n"
            "3. 回复应针对用户的需求\n"
            "4. 不要提及关键词或触发机制"
        )
        recent = _format_messages(context.recent_messages)
        memories = _format_memories(context.relevant_memories)
        user_info = _format_profile(context.user_profile, "用户")

        prompt = f"""## Bot 人设
{bot_persona or '一个友好的群聊助手'}

## 触发关键词
{match.keyword}

## 用户消息
{message}

## 近期对话
{recent or '（无）'}

## 相关记忆
{memories or '（无）'}

## 用户画像
{user_info or '（无）'}

## 回复方向
{direction}

请直接输出你的回复内容，不要包含任何前缀或标记。"""
        return system_prompt, prompt

    # ── 跟进计划 Prompt ──

    def build_followup_plan_prompt(
        self,
        bot_reply: str,
        context: AssembledContext,
    ) -> tuple[str, str]:
        system_prompt = (
            "你是一个对话跟进规划助手。Bot 刚刚回复了消息，请判断是否需要跟进。\n"
            "你必须只输出 JSON，不要输出任何其他内容。"
        )
        recent = _format_messages(context.recent_messages)
        memories = _format_memories(context.relevant_memories)

        prompt = f"""## Bot 回复
{bot_reply}

## 对话上下文
{recent or '（无）'}

## 相关记忆
{memories or '（无）'}

## 跟进要求
1. 判断 Bot 的回复是否需要后续补充
2. 判断用户是否可能需要进一步帮助
3. 跟进应自然、不突兀
4. 如果不需要跟进，should_followup 设为 false

输出 JSON：
{{
    "should_followup": true或false,
    "followup_type": "topic_extend或unfinished或emotional_care或confirmation",
    "delay_seconds": 30到120之间的整数,
    "direction": "跟进方向描述",
    "max_wait_messages": 3到8之间的整数
}}"""
        return system_prompt, prompt

    # ── 跟进回复生成 Prompt ──

    def build_followup_reply_prompt(
        self,
        plan: FollowupPlan,
        bot_reply: str,
        context: AssembledContext,
        bot_persona: str = "",
    ) -> tuple[str, str]:
        system_prompt = (
            "你是一个群聊助手，正在跟进之前的对话。请自然地继续对话。\n"
            "回复要求：\n"
            "1. 语气自然，像群聊参与者\n"
            "2. 不要重复之前说过的内容\n"
            "3. 跟进应自然衔接之前的对话\n"
            "4. 不要提及你是主动跟进的"
        )
        recent = _format_messages(context.recent_messages)
        memories = _format_memories(context.relevant_memories)

        prompt = f"""## Bot 人设
{bot_persona or '一个友好的群聊助手'}

## Bot 之前的回复
{bot_reply}

## 跟进类型
{plan.followup_type.value}

## 跟进方向
{plan.direction}

## 近期对话
{recent or '（无）'}

## 相关记忆
{memories or '（无）'}

请直接输出你的跟进回复内容，不要包含任何前缀或标记。"""
        return system_prompt, prompt

    # ── 动态关键词生成 Prompt ──

    def build_keyword_generation_prompt(
        self,
        context: AssembledContext,
        existing_keywords: list[str],
    ) -> tuple[str, str]:
        system_prompt = (
            "你是一个关键词提取助手。根据对话内容和用户画像，提取适合 Bot 主动回复的关键词。\n"
            "你必须只输出 JSON，不要输出任何其他内容。"
        )
        recent = _format_messages(context.recent_messages)
        user_info = _format_profile(context.user_profile, "用户")
        group_info = _format_profile(context.group_profile, "群聊")

        prompt = f"""## 近期对话
{recent or '（无）'}

## 用户画像
{user_info or '（无）'}

## 群聊画像
{group_info or '（无）'}

## 已有关键词
{', '.join(existing_keywords) or '（无）'}

## 提取要求
1. 提取用户可能需要帮助的话题关键词
2. 关键词应该是简短的词或短语（1-4个字）
3. 不要提取太宽泛的词（如"的"、"是"等）
4. 不要与已有关键词重复
5. 最多提取10个关键词

输出 JSON：
{{
    "keywords": ["关键词1", "关键词2", ...]
}}"""
        return system_prompt, prompt


def _format_messages(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return ""
    lines = []
    for msg in messages[-15:]:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        sender = msg.get("sender_name", msg.get("sender_id", ""))
        if role == "assistant":
            lines.append(f"[Bot]: {content}")
        elif sender:
            lines.append(f"[{sender}]: {content}")
        else:
            lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


def _format_memories(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return ""
    lines = []
    for mem in memories[:5]:
        content = mem.get("content", mem.get("summary", mem.get("fact", str(mem))))
        relevance = mem.get("relevance", "")
        if relevance:
            lines.append(f"- {content} (相关度: {relevance:.2f})")
        else:
            lines.append(f"- {content}")
    return "\n".join(lines)


def _format_profile(profile: dict[str, Any] | None, label: str) -> str:
    if not profile:
        return ""
    if "summary" in profile and profile["summary"]:
        return profile["summary"]
    parts = []
    for key, val in profile.items():
        if key in ("user_id", "group_id"):
            continue
        if val and str(val).strip():
            parts.append(f"{key}: {val}")
    return "\n".join(parts) if parts else ""
