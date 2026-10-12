"""Registro de auditoría local, persistente y con evidencia de manipulación.

Cada decisión del gateway (incluidas las rechazadas y los failsafe) queda en
DuckDB en el propio IPC, sin depender de red. Cada fila guarda el SHA-256 de su
contenido encadenado con el hash de la fila anterior: borrar o editar una fila
rompe la cadena desde ese punto y `verify_chain()` lo detecta.

DuckDB admite un solo proceso escritor; el gateway mantiene una conexión y
serializa las escrituras con un lock. Las llamadas bloqueantes van a un thread
para no frenar el event loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

GENESIS = "0" * 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    seq            BIGINT PRIMARY KEY,
    ts_utc         TIMESTAMP NOT NULL,
    request_id     VARCHAR NOT NULL,
    client_id      VARCHAR,
    asset_id       VARCHAR,
    event          VARCHAR NOT NULL,
    backend        VARCHAR,
    tier           VARCHAR,
    action         VARCHAR,
    target_tag     VARCHAR,
    new_value      DOUBLE,
    latency_ms     DOUBLE,
    rejected_reason VARCHAR,
    detail_json    VARCHAR,
    prev_hash      VARCHAR NOT NULL,
    row_hash       VARCHAR NOT NULL
)
"""

_FIELDS = (
    "seq", "ts_utc", "request_id", "client_id", "asset_id", "event", "backend", "tier",
    "action", "target_tag", "new_value", "latency_ms", "rejected_reason", "detail_json",
)


def _row_hash(prev_hash: str, row: dict[str, Any]) -> str:
    canonical = json.dumps({k: row[k] for k in _FIELDS}, sort_keys=True, default=str)
    return hashlib.sha256((prev_hash + canonical).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(path)
        self._con.execute(_SCHEMA)
        self._lock = threading.Lock()
        last = self._con.execute(
            "SELECT seq, row_hash FROM audit_log ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        self._seq, self._last_hash = (last[0], last[1]) if last else (0, GENESIS)

    def _write(self, event: str, request_id: str, **fields: Any) -> int:
        with self._lock:
            self._seq += 1
            row = {k: None for k in _FIELDS}
            row.update(
                seq=self._seq,
                ts_utc=datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0),
                request_id=request_id,
                event=event,
            )
            detail = fields.pop("detail", None)
            row.update({k: v for k, v in fields.items() if k in row})
            row["detail_json"] = json.dumps(detail, default=str) if detail is not None else None
            row_hash = _row_hash(self._last_hash, row)
            self._con.execute(
                f"INSERT INTO audit_log ({', '.join(_FIELDS)}, prev_hash, row_hash) "
                f"VALUES ({', '.join('?' * (len(_FIELDS) + 2))})",
                [row[k] for k in _FIELDS] + [self._last_hash, row_hash],
            )
            self._last_hash = row_hash
            return self._seq

    async def record(self, event: str, request_id: str, **fields: Any) -> int:
        return await asyncio.to_thread(self._write, event, request_id, **fields)

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._con.execute(
                "SELECT * FROM audit_log ORDER BY seq DESC LIMIT ?", [limit]
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def verify_chain(self) -> tuple[bool, int | None]:
        """(cadena íntegra, primer seq roto o None)."""
        with self._lock:
            cur = self._con.execute(
                f"SELECT {', '.join(_FIELDS)}, prev_hash, row_hash FROM audit_log ORDER BY seq"
            )
            rows = cur.fetchall()
        prev = GENESIS
        for r in rows:
            row = dict(zip(_FIELDS, r[: len(_FIELDS)]))
            if r[-2] != prev or _row_hash(prev, row) != r[-1]:
                return False, row["seq"]
            prev = r[-1]
        return True, None

    def close(self) -> None:
        with self._lock:
            self._con.close()
