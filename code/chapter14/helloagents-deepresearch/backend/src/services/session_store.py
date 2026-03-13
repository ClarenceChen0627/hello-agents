"""SQLite-backed storage for complete deep research sessions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class SqliteSessionStore:
    """Persist and retrieve complete research session snapshots."""

    def __init__(self, db_path: str) -> None:
        path = Path(db_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        self._db_path = path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS research_sessions (
                    session_id TEXT PRIMARY KEY,
                    note_id TEXT NOT NULL,
                    research_topic TEXT NOT NULL,
                    search_api TEXT NOT NULL DEFAULT '',
                    report_markdown TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    file_path TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL
                )
                """
            )

    def upsert_session(self, snapshot: dict[str, Any]) -> None:
        session_id = str(snapshot.get("session_id") or snapshot.get("note_id") or "").strip()
        if not session_id:
            raise ValueError("Session snapshot requires a stable session_id")

        note_id = str(snapshot.get("note_id") or session_id).strip()
        payload_json = json.dumps(snapshot, ensure_ascii=False)

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT created_at FROM research_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            created_at = str(snapshot.get("created_at") or "")
            if existing and existing["created_at"]:
                created_at = str(existing["created_at"])

            updated_at = str(snapshot.get("updated_at") or created_at)
            connection.execute(
                """
                INSERT INTO research_sessions (
                    session_id,
                    note_id,
                    research_topic,
                    search_api,
                    report_markdown,
                    created_at,
                    updated_at,
                    file_path,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    note_id = excluded.note_id,
                    research_topic = excluded.research_topic,
                    search_api = excluded.search_api,
                    report_markdown = excluded.report_markdown,
                    updated_at = excluded.updated_at,
                    file_path = excluded.file_path,
                    payload_json = excluded.payload_json
                """,
                (
                    session_id,
                    note_id,
                    str(snapshot.get("research_topic") or "").strip(),
                    str(snapshot.get("search_api") or "").strip(),
                    str(snapshot.get("report_markdown") or ""),
                    created_at,
                    updated_at,
                    str(snapshot.get("file_path") or "").strip(),
                    payload_json,
                ),
            )

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT session_id, note_id, research_topic, created_at, file_path
                FROM research_sessions
                ORDER BY datetime(created_at) DESC, rowid DESC
                """
            ).fetchall()

        return [
            {
                "session_id": row["session_id"],
                "note_id": row["note_id"],
                "research_topic": row["research_topic"],
                "created_at": row["created_at"],
                "file_path": row["file_path"],
            }
            for row in rows
        ]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM research_sessions WHERE session_id = ? OR note_id = ?",
                (session_id, session_id),
            ).fetchone()

        if not row:
            return None

        try:
            payload = json.loads(str(row["payload_json"]))
        except json.JSONDecodeError:
            return None

        return payload if isinstance(payload, dict) else None
