from __future__ import annotations

from typing import Any, Dict, List

from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register

from iris_reply.cooldown import parse_duration
from iris_reply.manager import ReplyManager


@register("iris_reply", "Cassia", "主动回复插件 - 信号检测与智能跟进", "1.0.0")
class IrisReplyPlugin(Star):
    def __init__(self, context: Context) -> None:
        super().__init__(context)
        config = self._load_config()
        self._manager = ReplyManager(context, config)
        self._bot_id: str = ""

    def _load_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        for key in (
            "proactive_enabled",
            "proactive_mode",
            "provider_id",
            "followup_enabled",
            "quiet_hours_enabled",
            "quiet_hours_start",
            "quiet_hours_end",
            "cooperation_context_enabled",
            "cooperation_profile_enabled",
            "cooperation_llm_confirm_enabled",
            "cooperation_llm_followup_enabled",
        ):
            try:
                value = self.context.get_config(key)
                if value is not None:
                    config[key] = value
            except Exception as e:
                logger.debug(f"Config load skip '{key}': {e}")
        return config

    async def initialize(self) -> None:
        await self._manager.initialize()
        try:
            platforms = self.context.get_platforms()
            for platform in platforms:
                if hasattr(platform, "bot") and hasattr(platform.bot, "self_id"):
                    self._bot_id = str(platform.bot.self_id)
                    break
        except Exception:
            pass
        logger.info("IrisReplyPlugin initialized")

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        '''监听群聊消息，用于信号检测和跟进回复'''
        try:
            text = event.message_str
            if not text:
                return

            if text.lstrip().startswith("/"):
                return

            message_obj = event.message_obj
            sender = message_obj.sender
            user_id = str(sender.user_id) if hasattr(sender, "user_id") else ""

            group_id = ""
            if hasattr(message_obj, "group_id") and message_obj.group_id:
                group_id = str(message_obj.group_id)
            if not group_id:
                return

            session_key = ""
            if hasattr(event, "session_id"):
                session_key = event.session_id or ""

            umo = event.unified_msg_origin
            self._manager.store_umo(group_id, umo)

            is_bot = user_id == self._bot_id

            sender_name = ""
            if hasattr(sender, "nickname"):
                sender_name = sender.nickname or ""
            elif hasattr(sender, "name"):
                sender_name = sender.name or ""

            await self._manager.on_message(
                text=text,
                user_id=user_id,
                group_id=group_id,
                session_key=session_key,
                is_bot=is_bot,
                sender_name=sender_name,
            )
        except Exception as e:
            logger.error(f"Error processing message in IrisReplyPlugin: {e}")

    @filter.command("iris_reply")
    async def cmd_iris_reply(self, event: AstrMessageEvent):
        '''Iris Reply 主动回复插件管理指令'''
        raw = event.message_str.strip()
        parts = [p for p in raw.split() if p and not p.startswith("/")]
        sub = parts[0].lower() if parts else ""

        if sub in ("", "help", "帮助"):
            yield event.plain_result(
                "Iris Reply 指令帮助:\n"
                "  /iris_reply status          - 查看插件运行状态\n"
                "  /iris_reply toggle [on|off] - 开关主动回复\n"
                "  /iris_reply mode [rule|hybrid] - 查看/切换决策模式\n"
                "  /iris_reply cooldown <on|off|status> [群号] [时长] - 冷却管理\n"
                "  /iris_reply followup [on|off]   - 开关跟进回复\n"
                "  /iris_reply quiet <on|off> [开始] [结束] - 静音时段\n"
                "\n"
                "冷却时长格式: 30m / 2h\n"
                "静音时间格式: HH:MM（如 23:00 07:00）"
            )
        elif sub == "status":
            yield event.plain_result(self._format_status())
        elif sub == "toggle":
            yield event.plain_result(self._handle_toggle(parts[1:]))
        elif sub == "mode":
            yield event.plain_result(self._handle_mode(parts[1:]))
        elif sub == "cooldown":
            yield event.plain_result(self._handle_cooldown(event, parts[1:]))
        elif sub == "followup":
            yield event.plain_result(self._handle_followup(parts[1:]))
        elif sub == "quiet":
            yield event.plain_result(self._handle_quiet(parts[1:]))
        else:
            yield event.plain_result(
                f"未知子命令: {sub}\n使用 /iris_reply help 查看帮助"
            )

    def _format_status(self) -> str:
        status = self._manager.status
        return (
            f"Iris Reply 状态:\n"
            f"  主动回复: {'✅ 开启' if status.get('proactive_enabled') else '❌ 关闭'}\n"
            f"  决策模式: {status.get('proactive_mode', 'rule')}\n"
            f"  跟进回复: {'✅ 开启' if status.get('followup_enabled') else '❌ 关闭'}\n"
            f"  静音时段: {'✅ 开启' if status.get('quiet_hours_enabled') else '❌ 关闭'}\n"
            f"  信号队列: {status.get('signal_queue_total', 0)} 个信号 / "
            f"{status.get('signal_queue_groups', 0)} 个群\n"
            f"  活跃调度器: {status.get('scheduler_active_groups', 0)} 个群\n"
            f"  冷却中: {status.get('cooldown_active_groups', 0)} 个群\n"
            f"  跟进窗口: {status.get('followup_active_expectations', 0)} 个"
        )

    def _handle_toggle(self, args: List[str]) -> str:
        if not args:
            current = self._manager.get_config("proactive_enabled", True)
            new_state = not current
            self._manager.update_config("proactive_enabled", new_state)
            return f"主动回复已{'开启' if new_state else '关闭'}"

        target = args[0].lower()
        if target in ("on", "开", "开启"):
            self._manager.update_config("proactive_enabled", True)
            return "主动回复已开启"
        elif target in ("off", "关", "关闭"):
            self._manager.update_config("proactive_enabled", False)
            return "主动回复已关闭"
        else:
            return "用法: /iris_reply toggle [on|off]"

    def _handle_mode(self, args: List[str]) -> str:
        if not args:
            current = self._manager.get_config("proactive_mode", "rule")
            return (
                f"当前模式: {current}\n可选模式: rule (纯规则), hybrid (规则+LLM确认)"
            )

        mode = args[0].lower()
        if mode in ("rule", "规则"):
            self._manager.update_config("proactive_mode", "rule")
            return "已切换到纯规则模式"
        elif mode in ("hybrid", "混合"):
            self._manager.update_config("proactive_mode", "hybrid")
            return "已切换到混合模式 (规则+LLM确认)"
        else:
            return "未知模式，可选: rule, hybrid"

    def _handle_cooldown(self, event: AstrMessageEvent, args: List[str]) -> str:
        if not args:
            return (
                "用法: /iris_reply cooldown <on|off|status> [群号] [时长]\n"
                "  on [群号] [时长] - 开启冷却（时长如 30m, 2h）\n"
                "  off [群号] - 关闭冷却\n"
                "  status [群号] - 查看冷却状态"
            )

        action = args[0].lower()
        group_id = args[1] if len(args) > 1 else self._get_current_group_id(event)

        if not group_id:
            return "无法确定群号，请指定群号"

        if action == "on":
            duration_str = args[2] if len(args) > 2 else None
            duration_minutes = parse_duration(duration_str) if duration_str else None
            return self._manager.activate_cooldown(
                group_id=group_id,
                duration_minutes=duration_minutes,
                initiated_by="user",
            )

        elif action == "off":
            return self._manager.deactivate_cooldown(group_id)

        elif action == "status":
            status = self._manager.get_cooldown_status(group_id)
            if status is None:
                return f"群 {group_id} 无冷却记录"
            active = "✅ 激活" if status["is_active"] else "❌ 未激活"
            return (
                f"群 {group_id} 冷却状态: {active}\n"
                f"开始时间: {status['started_at']}\n"
                f"到期时间: {status['expires_at']}\n"
                f"发起方: {status['initiated_by']}\n"
                f"原因: {status.get('reason', '无')}"
            )
        else:
            return "未知操作，请使用 on/off/status"

    def _handle_followup(self, args: List[str]) -> str:
        if not args:
            current = self._manager.get_config("followup_enabled", True)
            state_text = "开启" if current else "关闭"
            return (
                f"跟进回复当前: {state_text}\n用法: /iris_reply followup <on|off>"
            )

        target = args[0].lower()
        if target in ("on", "开", "开启"):
            self._manager.update_config("followup_enabled", True)
            return "跟进回复已开启"
        elif target in ("off", "关", "关闭"):
            self._manager.update_config("followup_enabled", False)
            return "跟进回复已关闭"
        else:
            return "用法: /iris_reply followup <on|off>"

    def _handle_quiet(self, args: List[str]) -> str:
        if not args:
            enabled = self._manager.get_config("quiet_hours_enabled", False)
            start = self._manager.get_config("quiet_hours_start", "23:00")
            end = self._manager.get_config("quiet_hours_end", "07:00")
            state_text = "开启" if enabled else "关闭"
            return (
                f"静音时段: {state_text}\n时段: {start} - {end}\n"
                "用法: /iris_reply quiet <on|off> [开始时间] [结束时间]\n"
                "示例: /iris_reply quiet on 23:00 07:00"
            )

        action = args[0].lower()
        if action in ("on", "开", "开启"):
            start = args[1] if len(args) > 1 else "23:00"
            end = args[2] if len(args) > 2 else "07:00"
            self._manager.update_config("quiet_hours_enabled", True)
            self._manager.update_config("quiet_hours_start", start)
            self._manager.update_config("quiet_hours_end", end)
            return f"静音时段已开启: {start} - {end}"
        elif action in ("off", "关", "关闭"):
            self._manager.update_config("quiet_hours_enabled", False)
            return "静音时段已关闭"
        else:
            return "用法: /iris_reply quiet <on|off> [开始] [结束]"

    def _get_current_group_id(self, event: AstrMessageEvent) -> str:
        try:
            message_obj = event.message_obj
            if hasattr(message_obj, "group_id") and message_obj.group_id:
                return str(message_obj.group_id)
        except Exception:
            pass
        return ""

    async def terminate(self):
        '''插件卸载/停用时调用，清理资源'''
        await self._manager.close()
        logger.info("IrisReplyPlugin terminated")
