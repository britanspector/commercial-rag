"""
commercial-rag FastAPI 服务入口。

用法：
    uvicorn rag_api:app --host 0.0.0.0 --port 8000 --app-dir src
    # 或
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --app-dir src
"""

from api.main import app

__all__ = ["app"]
