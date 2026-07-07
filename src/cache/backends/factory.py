"""L1/L2 后端工厂。"""

from __future__ import annotations

import logging

from cache.backends.base import ExactCacheBackend, SemanticCacheBackend
from cache.backends.memory import MemoryExactBackend
from cache.backends.milvus_semantic import MilvusSemanticBackend
from cache.backends.redis import RedisExactBackend
from cache.backends.semantic import NullSemanticBackend
from cache.config import CacheSettings, cache_settings

logger = logging.getLogger(__name__)


def create_exact_backend(settings: CacheSettings | None = None) -> ExactCacheBackend:
    cfg = settings or cache_settings

    if cfg.use_redis_l1:
        redis_backend = RedisExactBackend(
            url=cfg.redis_url,
            key_prefix=cfg.redis_key_prefix,
            timeout_s=cfg.redis_timeout_s,
            max_entries=cfg.max_entries,
        )
        if redis_backend.available:
            return redis_backend
        logger.warning(
            "Redis L1 不可用（%s），回退 MemoryExactBackend",
            redis_backend.last_error or "unknown",
        )

    return MemoryExactBackend(max_entries=cfg.max_entries)


def create_semantic_backend(settings: CacheSettings | None = None) -> SemanticCacheBackend:
    cfg = settings or cache_settings

    if cfg.use_milvus_l2:
        milvus_backend = MilvusSemanticBackend(
            db_path=cfg.l2_milvus_db_path,
            vector_dim=cfg.l2_vector_dim,
            search_top_k=cfg.l2_search_top_k,
            settings=cfg,
        )
        if milvus_backend.implemented:
            return milvus_backend
        logger.warning(
            "Milvus L2 不可用（%s），回退 NullSemanticBackend",
            milvus_backend.last_error or "unknown",
        )

    return NullSemanticBackend()
