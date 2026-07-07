"""
统一缓存管理器：编排 L1 精确缓存与 L2 语义缓存。

Pipeline / API 应通过 CacheManager 使用缓存，不直接访问后端存储。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from cache.backends.base import ExactCacheBackend, SemanticCacheBackend
from cache.backends.factory import create_exact_backend, create_semantic_backend
from cache.config import CacheSettings, cache_settings
from cache.policy import is_entry_expired, should_cache_result, ttl_for_entry
from cache.safety import is_stale_reject_reason, validate_entry_safety
from cache.stats import CacheStatsCollector
from cache.types import (
    CacheEntry,
    CacheInvalidateFilter,
    CacheInvalidateResult,
    CacheLayer,
    CacheLookupResult,
    CacheMetadataFilters,
    CacheQueryContext,
    CacheStats,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_default_manager: CacheManager | None = None


class CacheManager:
    """
    缓存门面：lookup → store → invalidate → stats。

    查询顺序：L1 精确 key →（可选）L2 语义近邻。
    命中后执行 policy 校验与 TTL 检查。
    """

    def __init__(
        self,
        *,
        exact_backend: ExactCacheBackend | None = None,
        semantic_backend: SemanticCacheBackend | None = None,
        settings: CacheSettings | None = None,
        stats: CacheStatsCollector | None = None,
    ) -> None:
        from cache.config import cache_settings as current_settings

        cfg = settings or current_settings
        self.settings = cfg
        self.exact_backend = exact_backend or create_exact_backend(cfg)
        self.semantic_backend = semantic_backend or create_semantic_backend(cfg)
        self.stats = stats or CacheStatsCollector()

    @property
    def active(self) -> bool:
        return self.settings.active

    def lookup(self, context: CacheQueryContext) -> CacheLookupResult:
        """查询缓存：先 L1，再 L2。"""
        if not self.active:
            logger.debug("cache lookup skipped: disabled scope=%s", context.key.scope.value)
            return CacheLookupResult.miss("cache_disabled")

        from cache.invalidate_hooks import ensure_fingerprint_sync

        ensure_fingerprint_sync()

        self.stats.record_lookup()
        key = context.key

        exact_entry = self.exact_backend.get(key.storage_key())
        if exact_entry is not None:
            validated = self._validate_entry(exact_entry, context, similarity=1.0)
            if validated.hit:
                self.stats.record_hit_l1()
                logger.info(
                    "cache L1 hit scope=%s query=%r stock=%s key_hash=%s",
                    key.scope.value,
                    key.query_normalized[:80],
                    key.stock_code or "-",
                    key.storage_key_hash()[:12],
                )
                return validated
            logger.info(
                "cache L1 reject scope=%s reason=%s key_hash=%s "
                "query_norm=%r company_hint=%s report_year=%s l2_enabled=%s",
                key.scope.value,
                validated.reject_reason or "policy",
                key.storage_key_hash()[:12],
                key.query_normalized[:80],
                (context.metadata_filters.company_name if context.metadata_filters else "") or "-",
                (context.metadata_filters.report_year if context.metadata_filters else "") or "-",
                self.semantic_backend.implemented,
            )
            if not validated.hit and is_stale_reject_reason(validated.reject_reason or ""):
                self._purge_stale_entry(exact_entry, validated.reject_reason or "stale")

        if context.query_embedding and self.semantic_backend.implemented:
            semantic_entry = self.semantic_backend.lookup(context)
            if semantic_entry is not None:
                semantic_entry.exact_match = False
                similarity = self._estimate_similarity(context, semantic_entry)
                validated = self._validate_entry(
                    semantic_entry,
                    context,
                    similarity=similarity,
                )
                if validated.hit:
                    self.stats.record_hit_l2()
                    logger.info(
                        "cache L2 hit scope=%s similarity=%.4f query=%r cached_query=%r",
                        key.scope.value,
                        validated.similarity or 0.0,
                        key.query_normalized[:60],
                        (semantic_entry.original_query or "")[:60],
                    )
                    return validated
                logger.info(
                    "cache L2 reject scope=%s reason=%s similarity=%.4f "
                    "query_norm=%r company_hint=%s report_year=%s threshold=%.2f",
                    key.scope.value,
                    validated.reject_reason or "policy",
                    similarity,
                    key.query_normalized[:80],
                    (context.metadata_filters.company_name if context.metadata_filters else "") or "-",
                    (context.metadata_filters.report_year if context.metadata_filters else "") or "-",
                    self.settings.similarity_threshold,
                )
                if is_stale_reject_reason(validated.reject_reason):
                    self._purge_stale_entry(semantic_entry, validated.reject_reason)

        self.stats.record_miss()
        self._refresh_entry_counts()
        logger.info(
            "cache miss scope=%s query=%r query_norm=%r stock=%s key_hash=%s "
            "company_hint=%s report_year=%s l2_enabled=%s has_embedding=%s",
            key.scope.value,
            (context.original_query or key.query_normalized)[:80],
            key.query_normalized[:80],
            key.stock_code or "-",
            key.storage_key_hash()[:12],
            (context.metadata_filters.company_name if context.metadata_filters else "") or "-",
            (context.metadata_filters.report_year if context.metadata_filters else "") or "-",
            self.semantic_backend.implemented,
            bool(context.query_embedding),
        )
        return CacheLookupResult.miss("not_found")

    def store(
        self,
        context: CacheQueryContext,
        *,
        payload: dict,
        refused: bool = False,
        top_rerank_score: float = 0.0,
        chunk_ids: list[str] | None = None,
        source_request_id: int | None = None,
        entry_metadata: CacheMetadataFilters | None = None,
    ) -> bool:
        """写入缓存（通过 policy 判断是否允许）。"""
        if not self.active:
            self.stats.record_store_skipped()
            return False

        key = context.key
        if not should_cache_result(
            scope=key.scope,
            refused=refused,
            query_type=key.query_type,
            settings=self.settings,
        ):
            self.stats.record_store_skipped()
            return False

        entry = CacheEntry(
            key=key,
            created_at_iso=datetime.now(timezone.utc).isoformat(),
            ttl_s=ttl_for_entry(
                scope=key.scope,
                refused=refused,
                query_type=key.query_type,
                settings=self.settings,
            ),
            query_embedding=context.query_embedding,
            payload=payload,
            chunk_ids=chunk_ids or [],
            refused=refused,
            top_rerank_score=top_rerank_score,
            exact_match=True,
            source_request_id=source_request_id,
            layer=CacheLayer.L1_EXACT,
            original_query=context.original_query or key.query_normalized,
            rewritten_query=context.rewritten_query or context.original_query or key.query_normalized,
            metadata_filters=entry_metadata or context.metadata_filters,
        )

        self.exact_backend.put(entry)

        if context.query_embedding and self.semantic_backend.implemented:
            semantic_entry = CacheEntry(
                key=key,
                created_at_iso=entry.created_at_iso,
                ttl_s=entry.ttl_s,
                query_embedding=context.query_embedding,
                payload=payload,
                chunk_ids=entry.chunk_ids,
                refused=refused,
                top_rerank_score=top_rerank_score,
                exact_match=False,
                source_request_id=source_request_id,
                layer=CacheLayer.L2_SEMANTIC,
                original_query=entry.original_query,
                rewritten_query=entry.rewritten_query,
                metadata_filters=entry.metadata_filters,
            )
            self.semantic_backend.put(semantic_entry)

        self.stats.record_store()
        self._refresh_entry_counts()
        logger.info(
            "cache store scope=%s ttl_s=%s key_hash=%s refused=%s backend=%s",
            key.scope.value,
            entry.ttl_s,
            key.storage_key_hash()[:12],
            refused,
            self.exact_backend.__class__.__name__,
        )
        return True

    def invalidate(self, filter_: CacheInvalidateFilter) -> CacheInvalidateResult:
        """按条件失效缓存条目。"""
        removed_exact = self.exact_backend.invalidate(filter_)
        removed_semantic = self.semantic_backend.invalidate(filter_)
        total = removed_exact + removed_semantic
        if total:
            self.stats.record_invalidation(total)
        self._refresh_entry_counts()
        return CacheInvalidateResult(
            removed_exact=removed_exact,
            removed_semantic=removed_semantic,
        )

    def invalidate_for_upload(self, doc_id: str) -> CacheInvalidateResult:
        """upload 成功后按 doc_id 失效相关 entry。"""
        return self.invalidate(CacheInvalidateFilter(doc_id=doc_id))

    def invalidate_all(self) -> CacheInvalidateResult:
        return self.invalidate(CacheInvalidateFilter(all_entries=True))

    def delete(self, context: CacheQueryContext) -> bool:
        """删除单条 L1 精确缓存。"""
        return self.exact_backend.delete(context.key.storage_key())

    def ping(self) -> dict:
        """检查 L1/L2 后端连通性（不抛异常）。"""
        return {
            "l1": {
                "backend": self.exact_backend.__class__.__name__,
                "ok": self.exact_backend.ping(),
                "available": getattr(self.exact_backend, "available", True),
                "last_error": getattr(self.exact_backend, "last_error", None),
            },
            "l2": {
                "backend": self.semantic_backend.__class__.__name__,
                "implemented": self.semantic_backend.implemented,
                "ok": self.semantic_backend.ping(),
                "available": getattr(self.semantic_backend, "available", False),
                "last_error": getattr(self.semantic_backend, "last_error", None),
            },
        }

    def stats_snapshot(self) -> CacheStats:
        self._refresh_entry_counts()
        return self.stats.snapshot()

    def reset_stats(self) -> None:
        self.stats.reset()
        self._refresh_entry_counts()

    def describe(self) -> dict:
        """运行状态摘要（供 /health 或前端监控）。"""
        snap = self.stats_snapshot()
        l1_backend = {
            "name": self.exact_backend.__class__.__name__,
            "available": getattr(self.exact_backend, "available", True),
            "last_error": getattr(self.exact_backend, "last_error", None),
        }
        return {
            "active": self.active,
            "settings": {
                "enabled": self.settings.enabled,
                "bypass": self.settings.bypass,
                "similarity_threshold": self.settings.similarity_threshold,
                "max_entries": self.settings.max_entries,
                "l1_backend": self.settings.l1_backend,
                "redis_configured": bool(self.settings.redis_url),
                "redis_key_prefix": self.settings.redis_key_prefix,
                "ttl_search_s": self.settings.ttl_search_s,
                "ttl_chat_s": self.settings.ttl_chat_s,
                "l2_backend": self.settings.l2_backend,
                "l2_milvus_db": self.settings.l2_milvus_db_path,
                "l2_search_top_k": self.settings.l2_search_top_k,
            },
            "backends": {
                "l1": l1_backend,
                "l2": {
                    "name": self.semantic_backend.__class__.__name__,
                    "implemented": self.semantic_backend.implemented,
                    "available": getattr(self.semantic_backend, "available", False),
                    "last_error": getattr(self.semantic_backend, "last_error", None),
                },
            },
            "stats": {
                "lookups": snap.lookups,
                "hits_l1": snap.hits_l1,
                "hits_l2": snap.hits_l2,
                "misses": snap.misses,
                "hit_rate": round(snap.hit_rate, 4),
                "requests": snap.requests,
                "l1_hit_rate": round(snap.l1_hit_rate, 4),
                "l2_hit_rate": round(snap.l2_hit_rate, 4),
                "total_hit_rate": round(snap.total_hit_rate, 4),
                "avg_latency_ms": round(snap.avg_latency_ms, 2),
                "avg_hit_latency_ms": round(snap.avg_hit_latency_ms, 2),
                "avg_miss_latency_ms": round(snap.avg_miss_latency_ms, 2),
                "avg_latency_saved_ms": round(snap.avg_latency_saved_ms, 2),
                "vector_retrievals_saved": snap.vector_retrievals_saved,
                "llm_calls_saved": snap.llm_calls_saved,
                "llm_call_reduction_rate": round(snap.llm_call_reduction_rate, 4),
                "safety_rejects": snap.safety_rejects,
                "stores": snap.stores,
                "exact_entries": snap.exact_entries,
                "semantic_entries": snap.semantic_entries,
            },
        }

    def _validate_entry(
        self,
        entry: CacheEntry,
        context: CacheQueryContext,
        *,
        similarity: float,
    ) -> CacheLookupResult:
        if is_entry_expired(entry):
            self.stats.record_reject_expired()
            return CacheLookupResult.miss("expired")

        ok, reason = validate_entry_safety(
            entry,
            context,
            similarity=similarity,
            settings=self.settings,
            check_chunks=True,
        )
        if not ok:
            self.stats.record_reject_policy()
            return CacheLookupResult.miss(reason)

        layer = CacheLayer.L2_SEMANTIC if not entry.exact_match else CacheLayer.L1_EXACT
        return CacheLookupResult.ok(entry, similarity=similarity, layer=layer)

    def _purge_stale_entry(self, entry: CacheEntry, reason: str) -> None:
        """校验失败的 stale entry 从 L1/L2 移除。"""
        storage_key = entry.key.storage_key()
        removed_exact = self.exact_backend.invalidate(
            CacheInvalidateFilter(storage_key_prefix=storage_key)
        )
        removed_semantic = 0
        if entry.cache_id and self.semantic_backend.implemented:
            delete_fn = getattr(self.semantic_backend, "delete_cache_id", None)
            if callable(delete_fn):
                removed_semantic = 1 if delete_fn(entry.cache_id) else 0
        if entry.chunk_ids and reason == "chunk_missing":
            for chunk_id in entry.chunk_ids:
                doc_id = chunk_id.rsplit("_c", 1)[0] if "_c" in chunk_id else chunk_id.split("_", 1)[0]
                if doc_id:
                    self.semantic_backend.invalidate(CacheInvalidateFilter(doc_id=doc_id))
        total = removed_exact + removed_semantic
        if total:
            self.stats.record_invalidation(total)
            logger.info("cache purged stale entry reason=%s removed=%s", reason, total)

    @staticmethod
    def _estimate_similarity(context: CacheQueryContext, entry: CacheEntry) -> float:
        if entry.semantic_similarity is not None:
            return float(entry.semantic_similarity)
        if not entry.query_embedding or not context.query_embedding:
            return 0.0
        import math

        dot = sum(a * b for a, b in zip(entry.query_embedding, context.query_embedding))
        norm_a = math.sqrt(sum(a * a for a in entry.query_embedding))
        norm_b = math.sqrt(sum(b * b for b in context.query_embedding))
        if norm_a <= 0 or norm_b <= 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _refresh_entry_counts(self) -> None:
        self.stats.set_entry_counts(
            exact=self.exact_backend.count(),
            semantic=self.semantic_backend.count(),
        )


def create_cache_manager(
    *,
    settings: CacheSettings | None = None,
    exact_backend: ExactCacheBackend | None = None,
    semantic_backend: SemanticCacheBackend | None = None,
) -> CacheManager:
    """工厂：创建 CacheManager（L1 Redis/内存 + L2 Milvus/占位）。"""
    from cache.config import cache_settings as current_settings

    cfg = settings or current_settings
    return CacheManager(
        settings=cfg,
        exact_backend=exact_backend,
        semantic_backend=semantic_backend,
    )


def get_cache_manager() -> CacheManager:
    """进程内单例（供 API 层懒加载）。"""
    global _default_manager
    if _default_manager is None:
        _default_manager = create_cache_manager()
    return _default_manager


def reset_cache_manager() -> None:
    """测试用：重置单例。"""
    global _default_manager
    _default_manager = None
