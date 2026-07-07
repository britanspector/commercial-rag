#!/usr/bin/env python3
"""
缓存存储层自测（L1 Redis/内存 + L2 Milvus）。

不依赖 /search / /chat API。用法：

    cd commercial-rag
    PYTHONPATH=src python -m cache.self_test
    PYTHONPATH=src python -m cache.self_test --redis-url redis://127.0.0.1:6379/0
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import time
from pathlib import Path

from cache import (
    CacheMetadataFilters,
    CacheScope,
    build_query_context,
    create_cache_manager,
    extract_chunk_ids,
    reset_cache_manager,
)
from cache.backends.memory import MemoryExactBackend
from cache.backends.milvus_semantic import MilvusSemanticBackend
from cache.backends.redis import RedisExactBackend
from cache.backends.semantic import NullSemanticBackend
from cache.config import CacheSettings
from rag_types import RAGQuery


def _unit_vector(dim: int, index: int = 0, noise_index: int | None = None, noise: float = 0.0) -> list[float]:
    vec = [0.0] * dim
    vec[index % dim] = 1.0
    if noise_index is not None and noise:
        vec[noise_index % dim] = noise
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


def _base_settings(**overrides) -> CacheSettings:
    defaults = dict(
        enabled=True,
        bypass=False,
        similarity_threshold=0.92,
        ttl_search_s=3600,
        ttl_chat_s=3600,
        ttl_refused_s=0,
        max_entries=1000,
        l1_backend="memory",
        redis_url="",
        redis_key_prefix="rag:selftest",
        redis_timeout_s=1.0,
        l2_backend="null",
        l2_milvus_db_path="",
        l2_vector_dim=1024,
        l2_search_top_k=5,
    )
    defaults.update(overrides)
    return CacheSettings(**defaults)


def _search_context(
    *,
    query: str,
    stock_code: str = "688008",
    index_fp: str = "idx-selftest-v1",
    embedding: list[float] | None = None,
    report_year: str = "2024",
    company_name: str = "澜起科技",
) -> object:
    rag_query = RAGQuery(query=query, stock_code=stock_code, query_type="factual")
    return build_query_context(
        scope=CacheScope.SEARCH,
        rag_query=rag_query,
        config_fingerprint="route=hybrid|rtk=30|rrk=5|ref=0.3500|hw=0.3500|pool=200",
        index_fingerprint=index_fp,
        query_embedding=embedding,
        original_query=query,
        rewritten_query=f"{query} {stock_code}".strip(),
        metadata_filters=CacheMetadataFilters(
            stock_code=stock_code,
            query_type="factual",
            company_name=company_name,
            report_year=report_year,
            doc_version=index_fp,
        ),
    )


def test_l1_memory_exact_hit() -> None:
    reset_cache_manager()
    settings = _base_settings(l1_backend="memory")
    mgr = create_cache_manager(
        settings=settings,
        exact_backend=MemoryExactBackend(max_entries=100),
    )
    assert mgr.ping()["l1"]["ok"] is True

    ctx = _search_context(query="澜起科技 2024 年 EPS 是多少")
    assert mgr.lookup(ctx).hit is False

    payload = {
        "query": "澜起科技 2024 年 EPS 是多少",
        "rerank": {"hits": [{"chunk_id": "doc688008_c1"}]},
    }
    assert mgr.store(ctx, payload=payload, chunk_ids=extract_chunk_ids(payload)) is True

    hit = mgr.lookup(ctx)
    assert hit.hit and hit.layer and hit.layer.value == "l1_exact"

    stats = mgr.stats_snapshot()
    assert stats.hits_l1 >= 1 and stats.stores >= 1
    print("[ok] L1 memory exact hit")


def test_l1_key_isolation() -> None:
    reset_cache_manager()
    mgr = create_cache_manager(
        settings=_base_settings(),
        exact_backend=MemoryExactBackend(max_entries=100),
    )
    ctx_a = _search_context(query="澜起科技 EPS", stock_code="688008")
    ctx_b = _search_context(query="澜起科技 EPS", stock_code="600000")
    payload = {"query": "x", "rerank": {"hits": [{"chunk_id": "c1"}]}}
    mgr.store(ctx_a, payload=payload, chunk_ids=["c1"])
    assert mgr.lookup(ctx_b).hit is False
    print("[ok] L1 different stock_code miss")


def test_l1_delete_and_invalidate() -> None:
    reset_cache_manager()
    backend = MemoryExactBackend(max_entries=100)
    mgr = create_cache_manager(settings=_base_settings(), exact_backend=backend)
    ctx = _search_context(query="删除测试")
    payload = {"query": "删除测试", "rerank": {"hits": [{"chunk_id": "d1_c1"}]}}
    mgr.store(ctx, payload=payload, chunk_ids=["d1_c1"])
    assert mgr.lookup(ctx).hit is True
    assert mgr.delete(ctx) is True
    assert mgr.lookup(ctx).hit is False

    mgr.store(ctx, payload=payload, chunk_ids=["d1_c1"])
    removed = mgr.invalidate_for_upload("d1")
    assert removed.total >= 1
    assert mgr.lookup(ctx).hit is False
    print("[ok] L1 delete / invalidate")


def test_l1_ttl() -> None:
    reset_cache_manager()
    settings = _base_settings(ttl_search_s=1)
    mgr = create_cache_manager(
        settings=settings,
        exact_backend=MemoryExactBackend(max_entries=100),
    )
    ctx = _search_context(query="TTL 测试")
    payload = {"query": "TTL 测试", "rerank": {"hits": [{"chunk_id": "t1_c1"}]}}
    mgr.store(ctx, payload=payload, chunk_ids=["t1_c1"])
    assert mgr.lookup(ctx).hit is True
    time.sleep(1.2)
    assert mgr.lookup(ctx).hit is False
    print("[ok] L1 TTL expire")


def test_l1_redis_or_skip(redis_url: str | None) -> None:
    if not redis_url:
        print("[skip] L1 Redis（未提供 --redis-url）")
        return

    reset_cache_manager()
    settings = _base_settings(
        l1_backend="redis",
        redis_url=redis_url,
        redis_key_prefix="rag:selftest",
    )
    backend = RedisExactBackend(
        url=redis_url,
        key_prefix=settings.redis_key_prefix,
        timeout_s=settings.redis_timeout_s,
        max_entries=settings.max_entries,
    )
    if not backend.available:
        print(f"[skip] L1 Redis 不可用: {backend.last_error}")
        return

    mgr = create_cache_manager(settings=settings, exact_backend=backend)
    assert mgr.ping()["l1"]["ok"] is True
    ctx = _search_context(query="Redis L1 精确命中测试")
    payload = {"query": ctx.key.query_normalized, "rerank": {"hits": [{"chunk_id": "r1_c1"}]}}
    mgr.invalidate_all()
    assert mgr.lookup(ctx).hit is False
    mgr.store(ctx, payload=payload, chunk_ids=["r1_c1"])
    assert mgr.lookup(ctx).hit is True
    print("[ok] L1 Redis exact hit")


def test_l1_redis_degrade() -> None:
    reset_cache_manager()
    settings = _base_settings(
        l1_backend="redis",
        redis_url="redis://127.0.0.1:59999/0",
        redis_timeout_s=0.2,
    )
    mgr = create_cache_manager(settings=settings)
    assert mgr.exact_backend.__class__.__name__ == "MemoryExactBackend"
    ctx = _search_context(query="降级测试")
    result = mgr.lookup(ctx)
    assert result.hit is False
    print("[ok] L1 Redis unavailable fallback")


def test_l2_milvus_semantic_hit() -> None:
    reset_cache_manager()
    dim = 1024
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "semantic_cache.db"
        settings = _base_settings(
            l2_backend="milvus",
            l2_milvus_db_path=str(db_path),
            l2_vector_dim=dim,
            similarity_threshold=0.90,
        )
        l2 = MilvusSemanticBackend(
            db_path=db_path,
            vector_dim=dim,
            search_top_k=5,
            settings=settings,
        )
        assert l2.implemented and l2.ping() is True

        mgr = create_cache_manager(
            settings=settings,
            exact_backend=MemoryExactBackend(max_entries=100),
            semantic_backend=l2,
        )

        vec_store = _unit_vector(dim, 0, 1, 0.05)
        vec_query = _unit_vector(dim, 0, 1, 0.04)

        ctx_store = _search_context(
            query="澜起科技 2024 年每股收益",
            embedding=vec_store,
        )
        payload = {
            "query": "澜起科技 2024 年每股收益",
            "rerank": {"hits": [{"chunk_id": "doc688008_c10"}]},
        }
        mgr.store(ctx_store, payload=payload, chunk_ids=["doc688008_c10"])

        ctx_lookup = _search_context(
            query="请告诉我澜起科技2024EPS",
            embedding=vec_query,
        )
        hit = mgr.lookup(ctx_lookup)
        assert hit.hit and hit.layer and hit.layer.value == "l2_semantic"
        assert hit.similarity is not None and hit.similarity >= 0.90
        print(f"[ok] L2 Milvus semantic hit similarity={hit.similarity:.4f}")


def test_l2_metadata_reject() -> None:
    reset_cache_manager()
    dim = 1024
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "semantic_cache.db"
        settings = _base_settings(
            l2_backend="milvus",
            l2_milvus_db_path=str(db_path),
            l2_vector_dim=dim,
            similarity_threshold=0.90,
        )
        l2 = MilvusSemanticBackend(db_path=db_path, vector_dim=dim, settings=settings)
        mgr = create_cache_manager(
            settings=settings,
            exact_backend=MemoryExactBackend(max_entries=100),
            semantic_backend=l2,
        )

        vec = _unit_vector(dim, 2, 3, 0.05)
        ctx_store = _search_context(query="澜起科技业绩", embedding=vec, report_year="2024")
        payload = {"query": "澜起科技业绩", "rerank": {"hits": [{"chunk_id": "doc688008_c2"}]}}
        mgr.store(ctx_store, payload=payload, chunk_ids=["doc688008_c2"])

        ctx_lookup = _search_context(
            query="澜起科技去年业绩怎么样",
            embedding=_unit_vector(dim, 2, 3, 0.04),
            report_year="2023",
        )
        assert mgr.lookup(ctx_lookup).hit is False
        print("[ok] L2 metadata report_year reject")


def test_l1_index_fingerprint_mismatch() -> None:
    reset_cache_manager()
    settings = _base_settings()
    mgr = create_cache_manager(
        settings=settings,
        exact_backend=MemoryExactBackend(max_entries=100),
        semantic_backend=NullSemanticBackend(),
    )
    ctx = _search_context(query="索引版本测试", index_fp="idx-v1")
    payload = {"query": "索引版本测试", "rerank": {"hits": [{"chunk_id": "idx_c1"}]}}
    mgr.store(ctx, payload=payload, chunk_ids=["idx_c1"])

    ctx_new = _search_context(query="索引版本测试", index_fp="idx-v2-changed")
    assert mgr.lookup(ctx_new).hit is False
    print("[ok] L1 index fingerprint mismatch reject")


def test_l1_stock_code_isolation() -> None:
    reset_cache_manager()
    mgr = create_cache_manager(
        settings=_base_settings(),
        exact_backend=MemoryExactBackend(max_entries=100),
        semantic_backend=NullSemanticBackend(),
    )
    ctx_a = _search_context(query="EPS", stock_code="688008", company_name="澜起科技")
    payload = {"query": "EPS", "rerank": {"hits": [{"chunk_id": "a_c1", "company_name": "澜起科技"}]}}
    mgr.store(ctx_a, payload=payload, chunk_ids=["a_c1"])
    ctx_b = _search_context(query="EPS", stock_code="600000", company_name="其他公司")
    assert mgr.lookup(ctx_b).hit is False
    print("[ok] L1 stock_code cross-company reject")


def test_safety_validate_entry() -> None:
    from cache.policy import validate_semantic_metadata
    from cache.safety import validate_entry_safety
    from cache.types import CacheEntry, CacheKey, CacheMetadataFilters, CacheQueryContext, CacheScope

    key = CacheKey(
        scope=CacheScope.SEARCH,
        query_normalized="test",
        stock_code="688008",
        query_type="factual",
        config_fingerprint="cfg",
        index_fingerprint="idx",
        metadata_filter_fingerprint="stock=688008|qtype=factual|co=澜起|yr=2024|doc=-|ver=idx",
    )
    entry = CacheEntry(
        key=key,
        created_at_iso="2026-06-05T12:00:00+00:00",
        ttl_s=3600,
        chunk_ids=["missing_chunk"],
        metadata_filters=CacheMetadataFilters(stock_code="688008", report_year="2024"),
    )
    ctx = CacheQueryContext(
        key=key,
        metadata_filters=CacheMetadataFilters(stock_code="688008", report_year="2023"),
    )
    ok, reason = validate_semantic_metadata(entry, ctx)
    assert not ok and reason == "report_year_mismatch"

    ctx2 = CacheQueryContext(
        key=key,
        metadata_filters=CacheMetadataFilters(stock_code="688008", report_year="2024"),
    )
    ok2, _ = validate_entry_safety(entry, ctx2, check_chunks=False)
    assert ok2
    print("[ok] safety validate_entry")


def test_telemetry_stats() -> None:
    from cache.stats import CacheStatsCollector
    from cache.telemetry import CacheRequestTelemetry, record_request_telemetry
    from cache.types import CacheScope

    stats = CacheStatsCollector()
    hit = CacheRequestTelemetry(
        scope=CacheScope.SEARCH,
        hit=True,
        source="l1_exact",
        similarity=1.0,
        reason="served",
        latency_ms=12.5,
        lookup_ms=2.0,
        pipeline_ms=0.0,
        vector_retrieval=False,
        llm_called=False,
    )
    record_request_telemetry(stats, hit)
    miss = CacheRequestTelemetry(
        scope=CacheScope.CHAT,
        hit=False,
        source="pipeline",
        reason="not_found",
        latency_ms=850.0,
        lookup_ms=5.0,
        pipeline_ms=840.0,
        vector_retrieval=True,
        llm_called=True,
    )
    record_request_telemetry(stats, miss)

    snap = stats.snapshot()
    assert snap.requests == 2
    assert snap.request_hits_l1 == 1
    assert snap.request_misses == 1
    assert snap.vector_retrievals_saved == 1
    assert snap.l1_hit_rate == 0.5
    assert snap.total_hit_rate == 0.5
    summary = snap.to_summary_dict()
    assert "avg_latency_ms" in summary
    print("[ok] telemetry stats")


def main() -> int:
    parser = argparse.ArgumentParser(description="缓存 L1/L2 存储层自测")
    parser.add_argument("--redis-url", default="", help="可选，测试 Redis L1")
    args = parser.parse_args()

    from cache.chunk_registry import register_test_chunk_ids

    register_test_chunk_ids(
        {
            "c1",
            "d1_c1",
            "t1_c1",
            "r1_c1",
            "idx_c1",
            "a_c1",
            "doc688008_c1",
            "doc688008_c10",
            "doc688008_c2",
            "missing_chunk",
        }
    )

    tests = [
        test_l1_memory_exact_hit,
        test_l1_key_isolation,
        test_l1_delete_and_invalidate,
        test_l1_ttl,
        test_l1_redis_degrade,
        test_l2_milvus_semantic_hit,
        test_l2_metadata_reject,
        test_l1_index_fingerprint_mismatch,
        test_l1_stock_code_isolation,
        test_safety_validate_entry,
        test_telemetry_stats,
    ]
    for fn in tests:
        fn()

    test_l1_redis_or_skip(args.redis_url.strip() or None)

    print("\n全部自测通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
