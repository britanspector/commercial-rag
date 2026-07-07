"""语义缓存类型定义。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CacheScope(str, Enum):
    SEARCH = "search"
    CHAT = "chat"


class CacheLayer(str, Enum):
    """缓存层级：L1 精确 key，L2 语义近邻。"""

    L1_EXACT = "l1_exact"
    L2_SEMANTIC = "l2_semantic"


@dataclass(frozen=True)
class CacheMetadataFilters:
    """L2 语义缓存 metadata 过滤与安全校验维度。"""

    stock_code: str = ""
    query_type: str = "factual"
    company_name: str = ""
    report_year: str = ""
    doc_id: str = ""
    doc_version: str = ""

    def fingerprint(self) -> str:
        return (
            f"stock={self.stock_code or '-'}|qtype={self.query_type or 'factual'}"
            f"|co={self.company_name or '-'}|yr={self.report_year or '-'}"
            f"|doc={self.doc_id or '-'}|ver={self.doc_version or '-'}"
        )


@dataclass(frozen=True)
class CacheKey:
    """精确命中用的逻辑 key（不含 query embedding）。"""

    scope: CacheScope
    query_normalized: str
    stock_code: str
    query_type: str
    config_fingerprint: str
    index_fingerprint: str
    generation_fingerprint: str = ""
    metadata_filter_fingerprint: str = ""

    def metadata_fingerprint(self) -> str:
        """检索 metadata 过滤维度（stock / query_type / 公司 / 年份 / 文档）。"""
        if self.metadata_filter_fingerprint:
            return self.metadata_filter_fingerprint
        return f"stock={self.stock_code or '-'}|qtype={self.query_type or 'factual'}"

    def storage_key(self) -> str:
        """
        L1 逻辑 key：scope + 检索配置 + 索引版本 + metadata + 规范化问题。

        各维度变化均导致不同 entry，避免错误命中。
        """
        parts = [
            f"scope={self.scope.value}",
            f"cfg={self.config_fingerprint}",
            f"idx={self.index_fingerprint}",
            f"gen={self.generation_fingerprint or '-'}",
            f"meta={self.metadata_fingerprint()}",
            f"q={self.query_normalized}",
        ]
        return "|".join(parts)

    def storage_key_hash(self) -> str:
        """稳定 SHA256 摘要，供 Redis 等存储使用。"""
        return hashlib.sha256(self.storage_key().encode("utf-8")).hexdigest()

    def redis_key(self, *, prefix: str = "rag:cache") -> str:
        """Redis String key（hash 避免特殊字符与过长 query）。"""
        base = prefix.rstrip(":")
        return f"{base}:l1:{self.storage_key_hash()}"

    def semantic_bucket_key(self) -> str:
        """L2 语义检索桶：同 scope + fingerprint + metadata 内做近邻搜索。"""
        return "|".join(
            [
                self.scope.value,
                self.config_fingerprint,
                self.index_fingerprint,
                self.generation_fingerprint or "-",
                self.metadata_fingerprint(),
            ]
        )

    def semantic_bucket_hash(self) -> str:
        return hashlib.sha256(self.semantic_bucket_key().encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class CacheQueryContext:
    """一次 lookup / store 的上下文（由 key_builder 构造）。"""

    key: CacheKey
    query_embedding: list[float] | None = None
    original_query: str = ""
    rewritten_query: str = ""
    metadata_filters: CacheMetadataFilters | None = None


@dataclass
class CacheEntry:
    """缓存条目（存储 Pipeline 输出 + 元数据）。"""

    key: CacheKey
    created_at_iso: str
    ttl_s: int
    query_embedding: list[float] | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    chunk_ids: list[str] = field(default_factory=list)
    refused: bool = False
    top_rerank_score: float = 0.0
    exact_match: bool = True
    source_request_id: int | None = None
    layer: CacheLayer = CacheLayer.L1_EXACT
    original_query: str = ""
    rewritten_query: str = ""
    metadata_filters: CacheMetadataFilters | None = None
    semantic_similarity: float | None = None
    cache_id: str = ""

    @property
    def entry_id(self) -> str:
        return self.key.storage_key()


@dataclass
class CacheLookupResult:
    hit: bool
    entry: CacheEntry | None = None
    similarity: float | None = None
    layer: CacheLayer | None = None
    reject_reason: str = ""

    @classmethod
    def miss(cls, reason: str = "") -> CacheLookupResult:
        return cls(hit=False, reject_reason=reason)

    @classmethod
    def ok(
        cls,
        entry: CacheEntry,
        *,
        similarity: float = 1.0,
        layer: CacheLayer = CacheLayer.L1_EXACT,
    ) -> CacheLookupResult:
        return cls(hit=True, entry=entry, similarity=similarity, layer=layer)


@dataclass
class CacheInvalidateFilter:
    """失效条件（各字段 None 表示不限制）。"""

    scope: CacheScope | None = None
    doc_id: str | None = None
    index_fingerprint: str | None = None
    storage_key_prefix: str | None = None
    all_entries: bool = False


@dataclass
class CacheInvalidateResult:
    removed_exact: int = 0
    removed_semantic: int = 0

    @property
    def total(self) -> int:
        return self.removed_exact + self.removed_semantic


@dataclass
class CacheStats:
    """累计统计快照。"""

    lookups: int = 0
    hits_l1: int = 0
    hits_l2: int = 0
    misses: int = 0
    stores: int = 0
    store_skipped: int = 0
    invalidations: int = 0
    reject_policy: int = 0
    reject_expired: int = 0
    exact_entries: int = 0
    semantic_entries: int = 0
    # 请求级（一次 run_search / run 计一条）
    requests: int = 0
    request_hits_l1: int = 0
    request_hits_l2: int = 0
    request_misses: int = 0
    request_bypass: int = 0
    request_search: int = 0
    request_chat: int = 0
    safety_rejects: int = 0
    vector_retrievals_saved: int = 0
    llm_calls_saved: int = 0
    total_latency_ms_sum: float = 0.0
    hit_latency_ms_sum: float = 0.0
    miss_latency_ms_sum: float = 0.0
    lookup_latency_ms_sum: float = 0.0
    pipeline_latency_ms_sum: float = 0.0

    @property
    def hits(self) -> int:
        return self.hits_l1 + self.hits_l2

    @property
    def hit_rate(self) -> float:
        if self.lookups == 0:
            return 0.0
        return self.hits / self.lookups

    @property
    def request_hits(self) -> int:
        return self.request_hits_l1 + self.request_hits_l2

    @property
    def l1_hit_rate(self) -> float:
        if self.requests == 0:
            return 0.0
        return self.request_hits_l1 / self.requests

    @property
    def l2_hit_rate(self) -> float:
        if self.requests == 0:
            return 0.0
        return self.request_hits_l2 / self.requests

    @property
    def total_hit_rate(self) -> float:
        if self.requests == 0:
            return 0.0
        return self.request_hits / self.requests

    @property
    def avg_latency_ms(self) -> float:
        if self.requests == 0:
            return 0.0
        return self.total_latency_ms_sum / self.requests

    @property
    def avg_hit_latency_ms(self) -> float:
        if self.request_hits == 0:
            return 0.0
        return self.hit_latency_ms_sum / self.request_hits

    @property
    def avg_miss_latency_ms(self) -> float:
        if self.request_misses == 0:
            return 0.0
        return self.miss_latency_ms_sum / self.request_misses

    @property
    def avg_latency_saved_ms(self) -> float:
        """命中相对未命中节省的平均延迟（miss_avg - hit_avg）。"""
        if self.request_hits == 0 or self.request_misses == 0:
            return 0.0
        return self.avg_miss_latency_ms - self.avg_hit_latency_ms

    @property
    def llm_call_reduction_rate(self) -> float:
        """chat 请求中因缓存少调 LLM 的比例。"""
        if self.request_chat == 0:
            return 0.0
        return self.llm_calls_saved / self.request_chat

    def to_summary_dict(self) -> dict:
        return {
            "lookups": self.lookups,
            "hits_l1": self.hits_l1,
            "hits_l2": self.hits_l2,
            "misses": self.misses,
            "lookup_hit_rate": round(self.hit_rate, 4),
            "requests": self.requests,
            "request_hits_l1": self.request_hits_l1,
            "request_hits_l2": self.request_hits_l2,
            "request_misses": self.request_misses,
            "request_bypass": self.request_bypass,
            "request_search": self.request_search,
            "request_chat": self.request_chat,
            "l1_hit_rate": round(self.l1_hit_rate, 4),
            "l2_hit_rate": round(self.l2_hit_rate, 4),
            "total_hit_rate": round(self.total_hit_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_hit_latency_ms": round(self.avg_hit_latency_ms, 2),
            "avg_miss_latency_ms": round(self.avg_miss_latency_ms, 2),
            "avg_latency_saved_ms": round(self.avg_latency_saved_ms, 2),
            "vector_retrievals_saved": self.vector_retrievals_saved,
            "llm_calls_saved": self.llm_calls_saved,
            "llm_call_reduction_rate": round(self.llm_call_reduction_rate, 4),
            "safety_rejects": self.safety_rejects,
            "stores": self.stores,
            "store_skipped": self.store_skipped,
            "invalidations": self.invalidations,
            "reject_policy": self.reject_policy,
            "reject_expired": self.reject_expired,
            "exact_entries": self.exact_entries,
            "semantic_entries": self.semantic_entries,
        }
