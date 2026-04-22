from __future__ import annotations

import logging
from typing import Any

from astrbot.api.star import Context
from astrbot.api.event import AstrMessageEvent

from iris_reply.storage.store import Store
from iris_reply.keyword.keyword_store import KeywordStore
from iris_reply.keyword.keyword_generator import KeywordGenerator
from iris_reply.models.models import GroupConfig

logger = logging.getLogger("iris_reply.commands")


class CommandHandler:
    def __init__(
        self,
        context: Context,
        store: Store,
        keyword_store: KeywordStore,
        keyword_generator: KeywordGenerator | None,
    ):
        self._context = context
        self._store = store
        self._keyword_store = keyword_store
        self._keyword_generator = keyword_generator

    async def handle(self, event: AstrMessageEvent, args: list[str]) -> str:
        if not args:
            return self._help()

        sub = args[0].lower()

        if sub == "proactive":
            return await self._handle_proactive(event, args[1:])
        elif sub == "followup":
            return await self._handle_followup(event, args[1:])
        elif sub == "keyword":
            return await self._handle_keyword(event, args[1:])
        elif sub == "cooldown":
            return await self._handle_cooldown(event, args[1:])
        elif sub == "status":
            return await self._handle_status(event)
        else:
            return f"未知子命令: {sub}\n" + self._help()

    def _help(self) -> str:
        return (
            "Iris Reply 主动回复插件\n"
            "命令列表:\n"
            "  /iris_reply proactive on|off|status - 开启/关闭/查看主动回复\n"
            "  /iris_reply followup on|off - 开启/关闭跟进回复\n"
            "  /iris_reply keyword add <词> - 添加静态关键词\n"
            "  /iris_reply keyword remove <词> - 移除静态关键词\n"
            "  /iris_reply keyword list - 列出所有关键词\n"
            "  /iris_reply keyword refresh - 刷新动态关键词\n"
            "  /iris_reply cooldown [秒数] - 查看/设置冷却时间\n"
            "  /iris_reply status - 查看当前状态"
        )

    async def _handle_proactive(self, event: AstrMessageEvent, args: list[str]) -> str:
        group_id = self._get_group_id(event)
        if not group_id:
            return "此命令仅在群聊中可用"

        if not args:
            return "用法: /iris_reply proactive on|off|status"

        action = args[0].lower()
        config = self._store.get_group_config(group_id)

        if action == "on":
            config.proactive_enabled = True
            self._store.set_group_config(config)
            return f"群 {group_id} 主动回复已开启"
        elif action == "off":
            config.proactive_enabled = False
            self._store.set_group_config(config)
            return f"群 {group_id} 主动回复已关闭"
        elif action == "status":
            status = "开启" if config.proactive_enabled else "关闭"
            return f"群 {group_id} 主动回复状态: {status}"
        else:
            return f"未知操作: {action}"

    async def _handle_followup(self, event: AstrMessageEvent, args: list[str]) -> str:
        group_id = self._get_group_id(event)
        if not group_id:
            return "此命令仅在群聊中可用"

        if not args:
            return "用法: /iris_reply followup on|off"

        action = args[0].lower()
        config = self._store.get_group_config(group_id)

        if action == "on":
            config.followup_enabled = True
            self._store.set_group_config(config)
            return f"群 {group_id} 跟进回复已开启"
        elif action == "off":
            config.followup_enabled = False
            self._store.set_group_config(config)
            return f"群 {group_id} 跟进回复已关闭"
        else:
            return f"未知操作: {action}"

    async def _handle_keyword(self, event: AstrMessageEvent, args: list[str]) -> str:
        group_id = self._get_group_id(event)
        if not group_id:
            return "此命令仅在群聊中可用"

        if not args:
            return "用法: /iris_reply keyword add|remove|list|refresh"

        action = args[0].lower()

        if action == "add":
            if len(args) < 2:
                return "用法: /iris_reply keyword add <关键词>"
            keyword = args[1]
            if self._keyword_store.add_static_keyword(group_id, keyword):
                return f"已添加关键词: {keyword}"
            return f"关键词已存在: {keyword}"

        elif action == "remove":
            if len(args) < 2:
                return "用法: /iris_reply keyword remove <关键词>"
            keyword = args[1]
            if self._keyword_store.remove_static_keyword(group_id, keyword):
                return f"已移除关键词: {keyword}"
            return f"关键词不存在: {keyword}"

        elif action == "list":
            static = self._keyword_store.get_static_keywords(group_id)
            dynamic = self._keyword_store.get_dynamic_keywords(group_id)
            lines = [f"静态关键词({len(static)}): {', '.join(static) if static else '无'}"]
            lines.append(f"动态关键词({len(dynamic)}): {', '.join(dynamic) if dynamic else '无'}")
            return "\n".join(lines)

        elif action == "refresh":
            if self._keyword_generator is None:
                return "动态关键词生成未启用"
            existing = self._keyword_store.get_all_keywords(group_id)
            keywords = await self._keyword_generator.refresh_all(group_id, existing)
            self._keyword_store.update_dynamic_keywords(group_id, keywords)
            kw_list = [kw for kw, _, _ in keywords]
            return f"动态关键词已刷新，共 {len(keywords)} 个: {', '.join(kw_list[:20])}"

        else:
            return f"未知操作: {action}"

    async def _handle_cooldown(self, event: AstrMessageEvent, args: list[str]) -> str:
        group_id = self._get_group_id(event)
        if not group_id:
            return "此命令仅在群聊中可用"

        config = self._store.get_group_config(group_id)

        if not args:
            return f"群 {group_id} 冷却时间: {config.cooldown_seconds} 秒"

        try:
            seconds = int(args[0])
            if seconds < 0:
                return "冷却时间不能为负数"
            config.cooldown_seconds = seconds
            self._store.set_group_config(config)
            return f"群 {group_id} 冷却时间已设置为 {seconds} 秒"
        except ValueError:
            return "请输入有效的秒数"

    async def _handle_status(self, event: AstrMessageEvent) -> str:
        group_id = self._get_group_id(event)
        if not group_id:
            return "此命令仅在群聊中可用"

        config = self._store.get_group_config(group_id)
        proactive = "开启" if config.proactive_enabled else "关闭"
        followup = "开启" if config.followup_enabled else "关闭"
        static_kws = self._keyword_store.get_static_keywords(group_id)
        dynamic_kws = self._keyword_store.get_dynamic_keywords(group_id)

        return (
            f"群 {group_id} 状态:\n"
            f"  主动回复: {proactive}\n"
            f"  跟进回复: {followup}\n"
            f"  冷却时间: {config.cooldown_seconds} 秒\n"
            f"  静态关键词: {len(static_kws)} 个\n"
            f"  动态关键词: {len(dynamic_kws)} 个"
        )

    def _get_group_id(self, event: AstrMessageEvent) -> str:
        try:
            gid = event.get_group_id()
            if gid:
                return str(gid)
        except Exception:
            pass
        return ""
