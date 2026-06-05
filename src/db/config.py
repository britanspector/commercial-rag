"""审计库连接配置：默认 SQLite，可通过环境变量切换 PostgreSQL。"""

from __future__ import annotations

import os
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data" / "audit" / "rag_audit.db"
DEFAULT_SQLITE_URL = f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"

# 环境变量（任选其一）：
#   RAG_DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/rag_audit
#   RAG_DATABASE_URL=sqlite:///E:/path/to/rag_audit.db
ENV_DATABASE_URL = "RAG_DATABASE_URL"
ENV_AUDIT_ENABLED = "RAG_AUDIT_ENABLED"  # 设为 0 / false 可关闭写入


def is_audit_enabled() -> bool:
    raw = os.environ.get(ENV_AUDIT_ENABLED, "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def get_database_url() -> str:
    explicit = os.environ.get(ENV_DATABASE_URL, "").strip()
    if explicit:
        return explicit
    DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DEFAULT_SQLITE_URL


def is_sqlite_url(url: str) -> bool:
    return url.startswith("sqlite:")
