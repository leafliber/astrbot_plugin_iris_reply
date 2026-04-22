from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from typing import Any

from iris_reply.models.models import (
    FollowupPlan,
    FollowupType,
    GroupConfig,
    KeywordSource,
)

logger = logging.getLogger("iris_reply.storage")


class Store:
    def __init__(self, data_dir: str):
        self._db_path = os.path.join(data_dir, "iris_reply.db")
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS group_config (
                group_id TEXT PRIMARY KEY,
                proactive_enabled INTEGER DEFAULT 0,
                followup_enabled INTEGER DEFAULT 0,
                cooldown_seconds INTEGER DEFAULT 300,
                created_at TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS static_keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                keyword TEXT NOT NULL,
                created_at TEXT,
                UNIQUE(group_id, keyword)
            );

            CREATE TABLE IF NOT EXISTS dynamic_keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                keyword TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                expires_at TEXT,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS reply_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                reply_type TEXT NOT NULL,
                triggered_at TEXT,
                keyword TEXT,
                confidence REAL
            );

            CREATE TABLE IF NOT EXISTS followup_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                plan_type TEXT NOT NULL,
                direction TEXT,
                delay_seconds INTEGER,
                max_wait_messages INTEGER,
                scheduled_at TEXT,
                status TEXT DEFAULT 'pending',
                created_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_static_keywords_group
                ON static_keywords(group_id);
            CREATE INDEX IF NOT EXISTS idx_dynamic_keywords_group
                ON dynamic_keywords(group_id);
            CREATE INDEX IF NOT EXISTS idx_reply_stats_group
                ON reply_stats(group_id);
            CREATE INDEX IF NOT EXISTS idx_followup_plans_group_status
                ON followup_plans(group_id, status);
        """)
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── Group Config ──

    def get_group_config(self, group_id: str) -> GroupConfig:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM group_config WHERE group_id = ?", (group_id,)
        ).fetchone()
        if row is None:
            return GroupConfig(group_id=group_id)
        return GroupConfig(
            group_id=row["group_id"],
            proactive_enabled=bool(row["proactive_enabled"]),
            followup_enabled=bool(row["followup_enabled"]),
            cooldown_seconds=row["cooldown_seconds"],
        )

    def set_group_config(self, config: GroupConfig) -> None:
        conn = self._get_conn()
        now = _now_iso()
        conn.execute(
            """INSERT INTO group_config (group_id, proactive_enabled, followup_enabled, cooldown_seconds, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(group_id) DO UPDATE SET
                   proactive_enabled=excluded.proactive_enabled,
                   followup_enabled=excluded.followup_enabled,
                   cooldown_seconds=excluded.cooldown_seconds,
                   updated_at=excluded.updated_at""",
            (config.group_id, int(config.proactive_enabled), int(config.followup_enabled),
             config.cooldown_seconds, now, now),
        )
        conn.commit()

    def get_all_enabled_groups(self) -> list[GroupConfig]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM group_config WHERE proactive_enabled = 1 OR followup_enabled = 1"
        ).fetchall()
        return [
            GroupConfig(
                group_id=r["group_id"],
                proactive_enabled=bool(r["proactive_enabled"]),
                followup_enabled=bool(r["followup_enabled"]),
                cooldown_seconds=r["cooldown_seconds"],
            )
            for r in rows
        ]

    # ── Static Keywords ──

    def get_static_keywords(self, group_id: str) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT keyword FROM static_keywords WHERE group_id = ?", (group_id,)
        ).fetchall()
        return [r["keyword"] for r in rows]

    def add_static_keyword(self, group_id: str, keyword: str) -> bool:
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO static_keywords (group_id, keyword, created_at) VALUES (?, ?, ?)",
                (group_id, keyword, _now_iso()),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def remove_static_keyword(self, group_id: str, keyword: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM static_keywords WHERE group_id = ? AND keyword = ?",
            (group_id, keyword),
        )
        conn.commit()
        return cursor.rowcount > 0

    # ── Dynamic Keywords ──

    def get_dynamic_keywords(self, group_id: str) -> list[str]:
        conn = self._get_conn()
        now = _now_iso()
        rows = conn.execute(
            "SELECT keyword FROM dynamic_keywords WHERE group_id = ? AND (expires_at IS NULL OR expires_at > ?)",
            (group_id, now),
        ).fetchall()
        return [r["keyword"] for r in rows]

    def set_dynamic_keywords(self, group_id: str, keywords: list[tuple[str, KeywordSource, float]]) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM dynamic_keywords WHERE group_id = ?", (group_id,))
        now = _now_iso()
        for kw, source, confidence in keywords:
            conn.execute(
                "INSERT INTO dynamic_keywords (group_id, keyword, source, confidence, created_at) VALUES (?, ?, ?, ?, ?)",
                (group_id, kw, source.value, confidence, now),
            )
        conn.commit()

    # ── Reply Stats ──

    def record_reply(self, group_id: str, reply_type: str, keyword: str | None = None, confidence: float = 0.0) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO reply_stats (group_id, reply_type, triggered_at, keyword, confidence) VALUES (?, ?, ?, ?, ?)",
            (group_id, reply_type, _now_iso(), keyword, confidence),
        )
        conn.commit()

    def get_reply_count_in_window(self, group_id: str, reply_type: str, window_seconds: int) -> int:
        conn = self._get_conn()
        cutoff = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - window_seconds))
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM reply_stats WHERE group_id = ? AND reply_type = ? AND triggered_at > ?",
            (group_id, reply_type, cutoff),
        ).fetchone()
        return row["cnt"] if row else 0

    # ── Followup Plans ──

    def save_followup_plan(self, group_id: str, plan: FollowupPlan) -> int:
        conn = self._get_conn()
        now = _now_iso()
        scheduled_at = time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() + plan.delay_seconds)
        )
        cursor = conn.execute(
            """INSERT INTO followup_plans (group_id, plan_type, direction, delay_seconds, max_wait_messages, scheduled_at, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)""",
            (group_id, plan.followup_type.value, plan.direction, plan.delay_seconds,
             plan.max_wait_messages, scheduled_at, now),
        )
        conn.commit()
        return cursor.lastrowid

    def cancel_pending_followups(self, group_id: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE followup_plans SET status = 'cancelled' WHERE group_id = ? AND status = 'pending'",
            (group_id,),
        )
        conn.commit()

    def get_pending_followup(self, group_id: str) -> dict[str, Any] | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM followup_plans WHERE group_id = ? AND status = 'pending' ORDER BY scheduled_at ASC LIMIT 1",
            (group_id,),
        ).fetchone()
        return dict(row) if row else None

    def update_followup_status(self, plan_id: int, status: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE followup_plans SET status = ? WHERE id = ?", (status, plan_id)
        )
        conn.commit()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
