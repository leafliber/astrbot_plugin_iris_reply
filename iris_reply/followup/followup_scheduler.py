from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Callable, Awaitable

from iris_reply.models.models import FollowupPlan
from iris_reply.storage.store import Store
from iris_reply.config.config import ReplyConfig

logger = logging.getLogger("iris_reply.followup_scheduler")


class FollowupScheduler:
    def __init__(self, store: Store, config: ReplyConfig):
        self._store = store
        self._config = config
        self._pending: dict[str, asyncio.Task] = {}
        self._message_counts: dict[str, int] = defaultdict(int)
        self._execute_callback: Callable[[str, FollowupPlan], Awaitable[None]] | None = None

    def set_execute_callback(
        self, callback: Callable[[str, FollowupPlan], Awaitable[None]]
    ) -> None:
        self._execute_callback = callback

    async def schedule(self, group_id: str, plan: FollowupPlan) -> None:
        await self.cancel(group_id)

        self._store.save_followup_plan(group_id, plan)
        self._message_counts[group_id] = 0

        task = asyncio.create_task(
            self._delayed_execute(group_id, plan)
        )
        self._pending[group_id] = task
        logger.info(
            "群 %s 跟进已调度，延迟 %d 秒，类型: %s",
            group_id, plan.delay_seconds, plan.followup_type.value,
        )

    async def cancel(self, group_id: str) -> None:
        task = self._pending.pop(group_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._store.cancel_pending_followups(group_id)
        self._message_counts.pop(group_id, None)

    def on_new_message(self, group_id: str) -> None:
        if group_id in self._pending:
            self._message_counts[group_id] += 1

    def should_cancel_on_message(self, group_id: str, plan: FollowupPlan | None) -> bool:
        if plan is None:
            return True
        count = self._message_counts.get(group_id, 0)
        return count >= plan.max_wait_messages

    async def _delayed_execute(self, group_id: str, plan: FollowupPlan) -> None:
        try:
            await asyncio.sleep(plan.delay_seconds)

            if self._execute_callback is None:
                logger.warning("跟进执行回调未设置")
                return

            await self._execute_callback(group_id, plan)

        except asyncio.CancelledError:
            logger.debug("群 %s 跟进已取消", group_id)
        except Exception as e:
            logger.error("群 %s 跟进执行失败: %s", group_id, e)
        finally:
            self._pending.pop(group_id, None)
            self._message_counts.pop(group_id, None)

    async def shutdown(self) -> None:
        for group_id, task in list(self._pending.items()):
            if not task.done():
                task.cancel()
        for group_id, task in self._pending.items():
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._pending.clear()
        self._message_counts.clear()
