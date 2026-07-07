"""语义缓存模块（Phase 2）：统一 CacheManager + L1/L2 后端抽象。"""

from cache.config import CacheSettings, cache_settings, load_cache_settings
from cache.key_builder import (
    build_cache_key,
    build_config_fingerprint_from_pipeline,
    build_generation_fingerprint_from_env,
    build_query_context,
    extract_chunk_ids,
)
from cache.manager import CacheManager, create_cache_manager, get_cache_manager, reset_cache_manager
from cache.policy import (
    build_config_fingerprint,
    build_generation_fingerprint,
    is_entry_expired,
    normalize_query,
    should_cache_result,
    should_serve_cached,
    ttl_for_entry,
    validate_semantic_metadata,
)
from cache.safety import validate_entry_safety, is_stale_reject_reason
from cache.telemetry import stats_summary_dict
from cache.types import (
    CacheEntry,
    CacheInvalidateFilter,
    CacheInvalidateResult,
    CacheKey,
    CacheLayer,
    CacheLookupResult,
    CacheMetadataFilters,
    CacheQueryContext,
    CacheScope,
    CacheStats,
)

__all__ = [
    "CacheManager",
    "CacheSettings",
    "CacheScope",
    "CacheLayer",
    "CacheKey",
    "CacheQueryContext",
    "CacheMetadataFilters",
    "CacheEntry",
    "CacheLookupResult",
    "CacheInvalidateFilter",
    "CacheInvalidateResult",
    "CacheStats",
    "cache_settings",
    "load_cache_settings",
    "create_cache_manager",
    "get_cache_manager",
    "reset_cache_manager",
    "build_cache_key",
    "build_query_context",
    "build_config_fingerprint_from_pipeline",
    "build_generation_fingerprint_from_env",
    "extract_chunk_ids",
    "build_config_fingerprint",
    "build_generation_fingerprint",
    "normalize_query",
    "should_cache_result",
    "should_serve_cached",
    "is_entry_expired",
    "ttl_for_entry",
    "validate_semantic_metadata",
    "validate_entry_safety",
    "is_stale_reject_reason",
    "stats_summary_dict",
]
