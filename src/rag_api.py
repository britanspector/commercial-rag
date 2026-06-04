"""
commercial-rag FastAPI 服务入口。

用法：
    uvicorn rag_api:app --host 0.0.0.0 --port 8000 --app-dir src
    # 或
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --app-dir src
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hf_env import bootstrap_hf_cache

bootstrap_hf_cache()

from api.main import app

__all__ = ["app"]
