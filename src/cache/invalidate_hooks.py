"""缓存失效钩子：知识库更新 / 文档 upload / 配置变化。"""

from __future__ import annotations

import logging

from cache.chunk_registry import reset_chunk_registry
from cache.index_fingerprint import compute_index_fingerprint
from cache.manager import get_cache_manager
from cache.types import CacheInvalidateFilter, CacheScope

logger = logging.getLogger(__name__)

_last_index_fingerprint: str | None = None
_last_generation_fingerprint: str | None = None


def ensure_fingerprint_sync() -> None:
    """
    进程启动或首次 lookup 前调用：索引或生成配置变化时失效全库。
    """
    global _last_index_fingerprint, _last_generation_fingerprint

    from cache.key_builder import build_generation_fingerprint_from_env

    current_index = compute_index_fingerprint()
    current_gen = build_generation_fingerprint_from_env()

    manager = get_cache_manager()
    if not manager.active:
        _last_index_fingerprint = current_index
        _last_generation_fingerprint = current_gen
        return

    invalidated = 0
    if _last_index_fingerprint is not None and _last_index_fingerprint != current_index:
        result = manager.invalidate(
            CacheInvalidateFilter(index_fingerprint=_last_index_fingerprint)
        )
        invalidated += result.total
        logger.info(
            "cache invalidated on index change old=%s new=%s removed=%s",
            _last_index_fingerprint[:48],
            current_index[:48],
            result.total,
        )

    if _last_generation_fingerprint is not None and _last_generation_fingerprint != current_gen:
        # chat 缓存依赖 generation fingerprint；search 条目 gen=- 不受影响
        result = manager.invalidate(CacheInvalidateFilter(scope=CacheScope.CHAT))
        invalidated += result.total
        logger.info(
            "cache invalidated on generation config change removed=%s fp=%s",
            result.total,
            current_gen,
        )

    _last_index_fingerprint = current_index
    _last_generation_fingerprint = current_gen

    if invalidated:
        logger.info("cache fingerprint sync total_removed=%s", invalidated)


def on_corpus_updated(*, doc_id: str = "") -> None:
    """
    upload / 索引重建后调用：按 doc_id 失效 + 刷新 chunk 注册表 + 索引指纹同步。
    """
    reset_chunk_registry()
    manager = get_cache_manager()
    if not manager.active:
        ensure_fingerprint_sync()
        return

    removed = 0
    if doc_id.strip():
        result = manager.invalidate_for_upload(doc_id.strip())
        removed += result.total
        logger.info("cache invalidated for doc_id=%s removed=%s", doc_id, result.total)

    ensure_fingerprint_sync()


def invalidate_all_caches() -> int:
    manager = get_cache_manager()
    result = manager.invalidate_all()
    reset_chunk_registry()
    ensure_fingerprint_sync()
    return result.total
