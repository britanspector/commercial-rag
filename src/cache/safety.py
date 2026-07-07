"""缓存命中安全校验：fingerprint / metadata / chunk 存在性。"""

from __future__ import annotations

from cache.chunk_registry import verify_chunks_exist
from cache.config import CacheSettings, cache_settings
from cache.policy import should_serve_cached, validate_semantic_metadata
from cache.types import CacheEntry, CacheQueryContext

# 命中后校验失败且 entry 已不可信 → 主动 purge
STALE_ENTRY_REASONS = frozenset(
    {
        "index_fingerprint_mismatch",
        "config_fingerprint_mismatch",
        "generation_fingerprint_mismatch",
        "chunk_missing",
        "metadata_missing",
        "stock_code_mismatch",
        "company_mismatch",
        "report_year_mismatch",
        "doc_id_mismatch",
        "doc_version_mismatch",
        "metadata_filter_mismatch",
        "query_type_mismatch",
        "expired",
    }
)


def validate_entry_safety(
    entry: CacheEntry,
    context: CacheQueryContext,
    *,
    similarity: float = 1.0,
    settings: CacheSettings | None = None,
    check_chunks: bool = True,
) -> tuple[bool, str]:
    """
    统一安全校验（L1 / L2 共用）。

    顺序：fingerprint → metadata filter → chunk 存在性。
    """
    meta_ok, meta_reason = validate_semantic_metadata(entry, context)
    if not meta_ok:
        return False, meta_reason

    ok, reason = should_serve_cached(
        entry,
        index_fingerprint=context.key.index_fingerprint,
        config_fingerprint=context.key.config_fingerprint,
        generation_fingerprint=context.key.generation_fingerprint,
        similarity=similarity,
        settings=settings,
    )
    if not ok:
        return False, reason

    if check_chunks and entry.chunk_ids:
        chunk_ok, chunk_reason, _missing = verify_chunks_exist(entry.chunk_ids)
        if not chunk_ok:
            return False, chunk_reason

    cfg = settings or cache_settings
    _ = cfg
    return True, ""


def is_stale_reject_reason(reason: str) -> bool:
    return reason in STALE_ENTRY_REASONS
