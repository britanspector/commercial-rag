"""语义缓存配置（环境变量，默认全部关闭）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_L2_MILVUS_DB = PROJECT_ROOT / "data" / "vector" / "semantic_cache" / "milvus.db"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


@dataclass(frozen=True)
class CacheSettings:
    enabled: bool
    bypass: bool
    similarity_threshold: float
    ttl_search_s: int
    ttl_chat_s: int
    ttl_refused_s: int
    max_entries: int
    l1_backend: str
    redis_url: str
    redis_key_prefix: str
    redis_timeout_s: float
    l2_backend: str
    l2_milvus_db_path: str
    l2_vector_dim: int
    l2_search_top_k: int

    @property
    def active(self) -> bool:
        return self.enabled and not self.bypass

    @property
    def use_redis_l1(self) -> bool:
        return self.l1_backend.strip().lower() == "redis" and bool(self.redis_url.strip())

    @property
    def use_milvus_l2(self) -> bool:
        return self.l2_backend.strip().lower() == "milvus"


def load_cache_settings() -> CacheSettings:
    """
    环境变量：
        RAG_SEMANTIC_CACHE_ENABLED=0|1     总开关（默认关）
        RAG_SEMANTIC_CACHE_BYPASS=0|1      强制 bypass（评测脚本用）
        RAG_SEMANTIC_CACHE_SIM_THRESHOLD   语义相似度阈值，默认 0.88
        RAG_SEMANTIC_CACHE_TTL_SEARCH_S    默认 86400
        RAG_SEMANTIC_CACHE_TTL_CHAT_S      默认 43200
        RAG_SEMANTIC_CACHE_TTL_REFUSED_S   默认 900（拒答极短 TTL；0=不缓存拒答）
        RAG_SEMANTIC_CACHE_MAX_ENTRIES     默认 10000
        RAG_SEMANTIC_CACHE_L1_BACKEND      memory | redis，默认 memory
        RAG_SEMANTIC_CACHE_REDIS_URL       如 redis://127.0.0.1:6379/0
        RAG_SEMANTIC_CACHE_REDIS_KEY_PREFIX 默认 rag:cache
        RAG_SEMANTIC_CACHE_REDIS_TIMEOUT_S 连接/读写超时，默认 1.0
        RAG_SEMANTIC_CACHE_L2_BACKEND      null | milvus，默认 milvus
        RAG_SEMANTIC_CACHE_L2_MILVUS_DB      默认 data/vector/semantic_cache/milvus.db
        RAG_SEMANTIC_CACHE_L2_VECTOR_DIM     默认 1024（与 bge-large-zh 一致）
        RAG_SEMANTIC_CACHE_L2_SEARCH_TOP_K   语义召回候选数，默认 5
    """
    l2_db_raw = os.environ.get("RAG_SEMANTIC_CACHE_L2_MILVUS_DB", "").strip()
    l2_milvus_db = l2_db_raw or str(DEFAULT_L2_MILVUS_DB)
    return CacheSettings(
        enabled=_env_bool("RAG_SEMANTIC_CACHE_ENABLED", False),
        bypass=_env_bool("RAG_SEMANTIC_CACHE_BYPASS", False),
        similarity_threshold=_env_float("RAG_SEMANTIC_CACHE_SIM_THRESHOLD", 0.88),
        ttl_search_s=_env_int("RAG_SEMANTIC_CACHE_TTL_SEARCH_S", 86400),
        ttl_chat_s=_env_int("RAG_SEMANTIC_CACHE_TTL_CHAT_S", 43200),
        ttl_refused_s=_env_int("RAG_SEMANTIC_CACHE_TTL_REFUSED_S", 900),
        max_entries=_env_int("RAG_SEMANTIC_CACHE_MAX_ENTRIES", 10000),
        l1_backend=os.environ.get("RAG_SEMANTIC_CACHE_L1_BACKEND", "memory").strip().lower(),
        redis_url=os.environ.get("RAG_SEMANTIC_CACHE_REDIS_URL", "").strip(),
        redis_key_prefix=os.environ.get("RAG_SEMANTIC_CACHE_REDIS_KEY_PREFIX", "rag:cache").strip()
        or "rag:cache",
        redis_timeout_s=_env_float("RAG_SEMANTIC_CACHE_REDIS_TIMEOUT_S", 1.0),
        l2_backend=os.environ.get("RAG_SEMANTIC_CACHE_L2_BACKEND", "milvus").strip().lower(),
        l2_milvus_db_path=l2_milvus_db,
        l2_vector_dim=_env_int("RAG_SEMANTIC_CACHE_L2_VECTOR_DIM", 1024),
        l2_search_top_k=_env_int("RAG_SEMANTIC_CACHE_L2_SEARCH_TOP_K", 5),
    )


cache_settings = load_cache_settings()
