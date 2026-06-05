"""初始化审计库表结构。

用法（项目根目录）：
    python -m db.init_db
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from db.config import get_database_url, is_audit_enabled
from db.engine import init_db


def main() -> None:
    if not is_audit_enabled():
        print("RAG_AUDIT_ENABLED=0，跳过建表")
        return
    init_db()
    print(f"审计库表已创建：{get_database_url()}")


if __name__ == "__main__":
    main()
