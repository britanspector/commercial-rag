"""数据库引擎与表初始化。"""

from __future__ import annotations

import threading

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from db.config import get_database_url, is_sqlite_url
from db.models import Base

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
_init_lock = threading.Lock()


def _sqlite_pragmas(dbapi_conn, _connection_record) -> None:
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_engine() -> Engine:
    global _engine, _session_factory
    with _init_lock:
        if _engine is None:
            url = get_database_url()
            connect_args = {"check_same_thread": False} if is_sqlite_url(url) else {}
            _engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
            if is_sqlite_url(url):
                event.listen(_engine, "connect", _sqlite_pragmas)
            _session_factory = sessionmaker(_engine, expire_on_commit=False)
        return _engine


def get_session_factory() -> sessionmaker[Session]:
    get_engine()
    assert _session_factory is not None
    return _session_factory


_CHAT_ANSWER_COLUMNS: dict[str, str] = {
    "refusal_message": "TEXT DEFAULT ''",
    "citation_count": "INTEGER DEFAULT 0",
    "evidence_check_json": "TEXT DEFAULT '{}'",
}

_REFUSAL_RECORD_COLUMNS: dict[str, str] = {
    "refusal_message": "TEXT DEFAULT ''",
    "evidence_check_json": "TEXT DEFAULT '{}'",
}


def _migrate_sqlite_columns(engine: Engine) -> None:
    """为已有 SQLite 库追加新列（幂等）。"""
    if not is_sqlite_url(str(engine.url)):
        return

    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    table_columns = {
        "chat_answers": {col["name"] for col in inspector.get_columns("chat_answers")}
        if inspector.has_table("chat_answers")
        else set(),
        "refusal_records": {col["name"] for col in inspector.get_columns("refusal_records")}
        if inspector.has_table("refusal_records")
        else set(),
    }

    with engine.begin() as conn:
        for column, ddl in _CHAT_ANSWER_COLUMNS.items():
            if column not in table_columns.get("chat_answers", set()):
                conn.execute(text(f"ALTER TABLE chat_answers ADD COLUMN {column} {ddl}"))
        for column, ddl in _REFUSAL_RECORD_COLUMNS.items():
            if column not in table_columns.get("refusal_records", set()):
                conn.execute(text(f"ALTER TABLE refusal_records ADD COLUMN {column} {ddl}"))


def init_db() -> None:
    """创建全部表（幂等），并迁移 SQLite 新列。"""
    engine = get_engine()
    Base.metadata.create_all(engine)
    _migrate_sqlite_columns(engine)


def db_status() -> dict[str, str | bool]:
    from db.config import is_audit_enabled

    if not is_audit_enabled():
        return {"enabled": False, "backend": "disabled", "url_masked": ""}

    url = get_database_url()
    backend = "sqlite" if is_sqlite_url(url) else "postgresql" if "postgres" in url else "other"
    return {
        "enabled": True,
        "backend": backend,
        "url_masked": _mask_url(url),
    }


def _mask_url(url: str) -> str:
    if "@" not in url:
        return url
    prefix, rest = url.split("://", 1)
    if "@" in rest:
        creds, host = rest.rsplit("@", 1)
        return f"{prefix}://***@{host}"
    return url
