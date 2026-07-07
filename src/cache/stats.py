"""线程安全的缓存统计收集器。"""

from __future__ import annotations

import threading

from cache.types import CacheScope, CacheStats


class CacheStatsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats = CacheStats()

    def record_lookup(self) -> None:
        with self._lock:
            self._stats.lookups += 1

    def record_hit_l1(self) -> None:
        with self._lock:
            self._stats.hits_l1 += 1

    def record_hit_l2(self) -> None:
        with self._lock:
            self._stats.hits_l2 += 1

    def record_miss(self) -> None:
        with self._lock:
            self._stats.misses += 1

    def record_store(self) -> None:
        with self._lock:
            self._stats.stores += 1

    def record_store_skipped(self) -> None:
        with self._lock:
            self._stats.store_skipped += 1

    def record_invalidation(self, removed: int) -> None:
        with self._lock:
            self._stats.invalidations += removed

    def record_reject_policy(self) -> None:
        with self._lock:
            self._stats.reject_policy += 1

    def record_reject_expired(self) -> None:
        with self._lock:
            self._stats.reject_expired += 1

    def record_request(
        self,
        *,
        hit: bool,
        source: str,
        latency_ms: float,
        lookup_ms: float,
        pipeline_ms: float,
        vector_retrieval: bool,
        llm_called: bool,
        safety_reject: bool,
        scope: CacheScope,
        bypass: bool = False,
    ) -> None:
        """记录一次 Pipeline 请求级遥测。"""
        with self._lock:
            self._stats.requests += 1
            if scope == CacheScope.SEARCH:
                self._stats.request_search += 1
            else:
                self._stats.request_chat += 1
            if bypass:
                self._stats.request_bypass += 1
            self._stats.total_latency_ms_sum += latency_ms
            self._stats.lookup_latency_ms_sum += lookup_ms
            if pipeline_ms > 0:
                self._stats.pipeline_latency_ms_sum += pipeline_ms

            if hit:
                self._stats.hit_latency_ms_sum += latency_ms
                if source == "l1_exact":
                    self._stats.request_hits_l1 += 1
                elif source == "l2_semantic":
                    self._stats.request_hits_l2 += 1
                if not vector_retrieval:
                    self._stats.vector_retrievals_saved += 1
                if scope == CacheScope.CHAT and not llm_called:
                    self._stats.llm_calls_saved += 1
            else:
                self._stats.request_misses += 1
                self._stats.miss_latency_ms_sum += latency_ms

            if safety_reject:
                self._stats.safety_rejects += 1

    def record_bypass(self) -> None:
        """兼容旧调用；新代码请用 record_request(bypass=True)。"""
        pass

    def set_entry_counts(self, *, exact: int, semantic: int) -> None:
        with self._lock:
            self._stats.exact_entries = exact
            self._stats.semantic_entries = semantic

    def snapshot(self) -> CacheStats:
        with self._lock:
            s = self._stats
            return CacheStats(
                lookups=s.lookups,
                hits_l1=s.hits_l1,
                hits_l2=s.hits_l2,
                misses=s.misses,
                stores=s.stores,
                store_skipped=s.store_skipped,
                invalidations=s.invalidations,
                reject_policy=s.reject_policy,
                reject_expired=s.reject_expired,
                exact_entries=s.exact_entries,
                semantic_entries=s.semantic_entries,
                requests=s.requests,
                request_hits_l1=s.request_hits_l1,
                request_hits_l2=s.request_hits_l2,
                request_misses=s.request_misses,
                request_bypass=s.request_bypass,
                request_search=s.request_search,
                request_chat=s.request_chat,
                safety_rejects=s.safety_rejects,
                vector_retrievals_saved=s.vector_retrievals_saved,
                llm_calls_saved=s.llm_calls_saved,
                total_latency_ms_sum=s.total_latency_ms_sum,
                hit_latency_ms_sum=s.hit_latency_ms_sum,
                miss_latency_ms_sum=s.miss_latency_ms_sum,
                lookup_latency_ms_sum=s.lookup_latency_ms_sum,
                pipeline_latency_ms_sum=s.pipeline_latency_ms_sum,
            )

    def reset(self) -> None:
        with self._lock:
            self._stats = CacheStats()
