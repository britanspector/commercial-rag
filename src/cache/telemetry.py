"""缓存请求遥测：结构化日志与累计统计。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cache.safety import is_stale_reject_reason
from cache.stats import CacheStatsCollector
from cache.types import CacheLayer, CacheLookupResult, CacheScope

if TYPE_CHECKING:
    from cache.manager import CacheManager

logger = logging.getLogger(__name__)


@dataclass
class CacheLookupAttempt:
    """单次 lookup 尝试（L1 或 L2）。"""

    layer: str
    hit: bool
    latency_ms: float = 0.0
    similarity: float | None = None
    safety_ok: bool = True
    safety_reason: str = ""


@dataclass
class CacheRequestTelemetry:
    """单次 Pipeline 请求的缓存遥测快照。"""

    scope: CacheScope
    hit: bool = False
    source: str = "pipeline"  # l1_exact | l2_semantic | pipeline | none
    similarity: float | None = None
    reason: str = ""
    safety_ok: bool = True
    safety_reason: str = ""
    latency_ms: float = 0.0
    lookup_ms: float = 0.0
    pipeline_ms: float = 0.0
    vector_retrieval: bool = False
    llm_called: bool = False
    cache_enabled: bool = True
    cache_bypass: bool = False
    attempts: list[CacheLookupAttempt] = field(default_factory=list)
    stock_code: str = ""
    query_type: str = "factual"

    def to_dict(self) -> dict:
        return {
            "scope": self.scope.value,
            "hit": self.hit,
            "source": self.source,
            "similarity": self.similarity,
            "reason": self.reason,
            "safety_ok": self.safety_ok,
            "safety_reason": self.safety_reason,
            "latency_ms": round(self.latency_ms, 2),
            "lookup_ms": round(self.lookup_ms, 2),
            "pipeline_ms": round(self.pipeline_ms, 2),
            "vector_retrieval": self.vector_retrieval,
            "llm_called": self.llm_called,
            "cache_enabled": self.cache_enabled,
            "cache_bypass": self.cache_bypass,
            "attempts": [
                {
                    "layer": a.layer,
                    "hit": a.hit,
                    "latency_ms": round(a.latency_ms, 2),
                    "similarity": a.similarity,
                    "safety_ok": a.safety_ok,
                    "safety_reason": a.safety_reason,
                }
                for a in self.attempts
            ],
            "stock_code": self.stock_code or "-",
            "query_type": self.query_type,
        }


class CacheRequestTimer:
    """轻量计时上下文。"""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0


def _layer_name(layer: CacheLayer | None) -> str:
    return layer.value if layer else "pipeline"


def attempt_from_lookup(lookup: CacheLookupResult, *, latency_ms: float, layer: str) -> CacheLookupAttempt:
    if lookup.hit:
        return CacheLookupAttempt(
            layer=layer,
            hit=True,
            latency_ms=latency_ms,
            similarity=lookup.similarity,
            safety_ok=True,
        )
    reason = lookup.reject_reason or "not_found"
    stale = is_stale_reject_reason(reason)
    return CacheLookupAttempt(
        layer=layer,
        hit=False,
        latency_ms=latency_ms,
        similarity=lookup.similarity,
        safety_ok=not stale,
        safety_reason=reason if stale else "",
    )


def finalize_telemetry(
    telemetry: CacheRequestTelemetry,
    *,
    lookup: CacheLookupResult | None,
    lookup_ms: float,
    pipeline_ms: float,
    total_ms: float,
) -> CacheRequestTelemetry:
    """根据最终 lookup 结果补全遥测字段。"""
    telemetry.lookup_ms = lookup_ms
    telemetry.pipeline_ms = pipeline_ms
    telemetry.latency_ms = total_ms

    if lookup is not None and lookup.hit:
        telemetry.hit = True
        telemetry.source = _layer_name(lookup.layer)
        telemetry.similarity = lookup.similarity
        telemetry.reason = "served"
        telemetry.safety_ok = True
        telemetry.vector_retrieval = False
        telemetry.llm_called = False
        return telemetry

    if lookup is not None and lookup.reject_reason:
        telemetry.reason = lookup.reject_reason
        if is_stale_reject_reason(lookup.reject_reason):
            telemetry.safety_ok = False
            telemetry.safety_reason = lookup.reject_reason

    telemetry.hit = False
    telemetry.source = "pipeline"
    telemetry.vector_retrieval = True
    telemetry.llm_called = telemetry.scope == CacheScope.CHAT
    if not telemetry.reason:
        telemetry.reason = "not_found"
    return telemetry


def record_request_telemetry(
    stats: CacheStatsCollector,
    telemetry: CacheRequestTelemetry,
) -> None:
    """写入累计统计。"""
    stats.record_request(
        hit=telemetry.hit,
        source=telemetry.source,
        latency_ms=telemetry.latency_ms,
        lookup_ms=telemetry.lookup_ms,
        pipeline_ms=telemetry.pipeline_ms,
        vector_retrieval=telemetry.vector_retrieval,
        llm_called=telemetry.llm_called,
        safety_reject=not telemetry.safety_ok and bool(telemetry.safety_reason),
        scope=telemetry.scope,
        bypass=telemetry.cache_bypass or telemetry.reason in ("cache_bypass", "cache_disabled"),
    )


def log_request_telemetry(telemetry: CacheRequestTelemetry, *, query_preview: str = "") -> None:
    """结构化 INFO 日志，便于日志聚合与排查。"""
    preview = (query_preview or "")[:80]
    logger.info(
        "cache request scope=%s hit=%s source=%s similarity=%s safety_ok=%s "
        "reason=%s latency_ms=%.1f lookup_ms=%.1f pipeline_ms=%.1f "
        "vector_retrieval=%s llm_called=%s stock=%s query=%r",
        telemetry.scope.value,
        telemetry.hit,
        telemetry.source,
        f"{telemetry.similarity:.4f}" if telemetry.similarity is not None else "-",
        telemetry.safety_ok,
        telemetry.reason or "-",
        telemetry.latency_ms,
        telemetry.lookup_ms,
        telemetry.pipeline_ms,
        telemetry.vector_retrieval,
        telemetry.llm_called,
        telemetry.stock_code or "-",
        preview,
    )


def emit_request_telemetry(
    manager: CacheManager,
    telemetry: CacheRequestTelemetry,
    *,
    query_preview: str = "",
) -> None:
    record_request_telemetry(manager.stats, telemetry)
    log_request_telemetry(telemetry, query_preview=query_preview)


def stats_summary_dict(stats_collector: CacheStatsCollector) -> dict:
    """供 API / describe 使用的统计摘要。"""
    snap = stats_collector.snapshot()
    return snap.to_summary_dict()
