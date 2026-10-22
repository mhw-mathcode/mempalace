from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from .media import MediaRef
from .schemas import AddRequest


class IdempotencyConflict(RuntimeError):
    """A request ID was reused with a different canonical payload."""


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    request_id: str
    user_id: str
    session_id: str
    content: Any
    created_at: str
    state: str
    child_ids: tuple[str, ...]
    media_refs: tuple[MediaRef, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def canonical_payload_digest(request: AddRequest) -> str:
    payload = _canonical_json(request).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def memory_id_for_request(request_id: str) -> str:
    return f"mem_{hashlib.sha256(request_id.encode('utf-8')).hexdigest()}"


def child_id_for(memory_id: str, kind: str, ordinal: int) -> str:
    source = f"{memory_id}\0{kind}\0{ordinal}".encode()
    return f"vec_{hashlib.sha256(source).hexdigest()}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class StateStore:
    def __init__(self, path: Path, busy_timeout_ms: int = 5_000):
        self.path = Path(path)
        self.busy_timeout_ms = busy_timeout_ms
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_ms)}")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    request_id TEXT PRIMARY KEY,
                    payload_digest TEXT NOT NULL,
                    memory_id TEXT NOT NULL UNIQUE,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('pending', 'committed'))
                );

                CREATE INDEX IF NOT EXISTS memories_user_state
                    ON memories(user_id, state);

                CREATE TABLE IF NOT EXISTS vector_children (
                    child_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    FOREIGN KEY(memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
                    UNIQUE(memory_id, ordinal)
                );

                CREATE TABLE IF NOT EXISTS media_refs (
                    memory_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    suffix TEXT NOT NULL,
                    byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
                    PRIMARY KEY(memory_id, ordinal),
                    FOREIGN KEY(memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE
                );
                """
            )

    def begin_add(
        self,
        request: AddRequest,
        content: Any,
        child_ids: Sequence[str],
        media_refs: Sequence[MediaRef],
    ) -> tuple[MemoryRecord, Literal["created", "pending", "committed"]]:
        digest = canonical_payload_digest(request)
        memory_id = memory_id_for_request(request.request_id)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT * FROM memories WHERE request_id = ?", (request.request_id,)
                ).fetchone()
                if existing is not None:
                    if existing["payload_digest"] != digest:
                        raise IdempotencyConflict(
                            "request_id already exists with a different payload"
                        )
                    record = self._record_from_row(connection, existing)
                    connection.commit()
                    return record, record.state  # type: ignore[return-value]

                connection.execute(
                    """
                    INSERT INTO memories (
                        request_id, payload_digest, memory_id, user_id, session_id,
                        payload_json, content_json, created_at, state
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                    """,
                    (
                        request.request_id,
                        digest,
                        memory_id,
                        request.user_id,
                        request.session_id,
                        _canonical_json(request),
                        _canonical_json(content),
                        _utc_now(),
                    ),
                )
                connection.executemany(
                    "INSERT INTO vector_children (child_id, memory_id, ordinal) VALUES (?, ?, ?)",
                    [(child_id, memory_id, index) for index, child_id in enumerate(child_ids)],
                )
                connection.executemany(
                    """
                    INSERT INTO media_refs (
                        memory_id, ordinal, sha256, mime_type, suffix, byte_size
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            memory_id,
                            index,
                            ref.sha256,
                            ref.mime_type,
                            ref.suffix,
                            ref.byte_size,
                        )
                        for index, ref in enumerate(media_refs)
                    ],
                )
                row = connection.execute(
                    "SELECT * FROM memories WHERE request_id = ?", (request.request_id,)
                ).fetchone()
                record = self._record_from_row(connection, row)
                connection.commit()
                return record, "created"
            except Exception:
                connection.rollback()
                raise

    def mark_committed(self, request_id: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE memories SET state = 'committed' WHERE request_id = ?",
                (request_id,),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise KeyError(request_id)
            connection.commit()

    def get_by_request_id(self, request_id: str) -> MemoryRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memories WHERE request_id = ?", (request_id,)
            ).fetchone()
            return None if row is None else self._record_from_row(connection, row)

    def load_add_request(self, request_id: str) -> AddRequest | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM memories WHERE request_id = ?", (request_id,)
            ).fetchone()
        if row is None:
            return None
        return AddRequest.model_validate_json(row["payload_json"])

    def hydrate_committed(
        self, ids: Sequence[str], user_id: str
    ) -> dict[str, MemoryRecord]:
        if not ids:
            return {}
        unique_ids = list(dict.fromkeys(ids))
        placeholders = ",".join("?" for _ in unique_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM memories
                WHERE memory_id IN ({placeholders})
                  AND user_id = ?
                  AND state = 'committed'
                """,
                (*unique_ids, user_id),
            ).fetchall()
            by_id = {
                row["memory_id"]: self._record_from_row(connection, row) for row in rows
            }
        return {memory_id: by_id[memory_id] for memory_id in unique_ids if memory_id in by_id}

    def _record_from_row(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> MemoryRecord:
        child_ids = tuple(
            child["child_id"]
            for child in connection.execute(
                "SELECT child_id FROM vector_children WHERE memory_id = ? ORDER BY ordinal",
                (row["memory_id"],),
            ).fetchall()
        )
        media_refs = tuple(
            MediaRef(
                sha256=media["sha256"],
                mime_type=media["mime_type"],
                suffix=media["suffix"],
                byte_size=media["byte_size"],
            )
            for media in connection.execute(
                """
                SELECT sha256, mime_type, suffix, byte_size
                FROM media_refs WHERE memory_id = ? ORDER BY ordinal
                """,
                (row["memory_id"],),
            ).fetchall()
        )
        return MemoryRecord(
            id=row["memory_id"],
            request_id=row["request_id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            content=json.loads(row["content_json"]),
            created_at=row["created_at"],
            state=row["state"],
            child_ids=child_ids,
            media_refs=media_refs,
        )
