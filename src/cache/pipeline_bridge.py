"""CacheManager ↔ RAG Pipeline 桥接：上下文构建、命中还原、写入缓存。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cache import (
    CacheMetadataFilters,
    CacheScope,
    build_config_fingerprint_from_pipeline,
    build_generation_fingerprint_from_env,
    build_query_context,
    extract_chunk_ids,
    get_cache_manager,
)
from cache.config import cache_settings
from cache.index_fingerprint import compute_index_fingerprint
from cache.manager import CacheManager
from cache.metadata_builder import build_metadata_filters, enrich_metadata_from_result
from cache.metadata_extract import extract_report_year
from cache.telemetry import (
    CacheLookupAttempt,
    CacheRequestTelemetry,
    CacheRequestTimer,
    attempt_from_lookup,
    emit_request_telemetry,
    finalize_telemetry,
)
from cache.types import CacheLookupResult, CacheQueryContext
from rag_types import (
    CacheInfo,
    Citation,
    EvidenceCheckResult,
    HybridRetrieveResult,
    QueryRewriteResult,
    RAGPipelineResult,
    RAGQuery,
    RAGSearchResult,
    RetrievedChunk,
    RerankStepResult,
)

if TYPE_CHECKING:
    from rag_pipeline import RAGPipeline, RAGPipelineConfig


def cache_enabled(*, use_cache: bool) -> bool:
    if not use_cache:
        return False
    return get_cache_manager().settings.active


def build_cache_context(
    rag_query: RAGQuery,
    config: RAGPipelineConfig,
    *,
    scope: CacheScope,
    rewrite: QueryRewriteResult | None = None,
    result: RAGSearchResult | RAGPipelineResult | None = None,
) -> CacheQueryContext:
    index_fp = compute_index_fingerprint()
    generation_fp = (
        build_generation_fingerprint_from_env()
        if scope == CacheScope.CHAT
        else ""
    )
    metadata = build_metadata_filters(rag_query, index_fingerprint=index_fp)

    return build_query_context(
        scope=scope,
        rag_query=rag_query,
        config_fingerprint=build_config_fingerprint_from_pipeline(config),
        index_fingerprint=index_fp,
        generation_fingerprint=generation_fp,
        query_embedding=rewrite.query_vector if rewrite else None,
        original_query=rewrite.original_query if rewrite else rag_query.query,
        rewritten_query=rewrite.query if rewrite else rag_query.query,
        metadata_filters=metadata,
    )


def entry_metadata_from_result(
    rag_query: RAGQuery,
    result: RAGSearchResult | RAGPipelineResult,
) -> CacheMetadataFilters:
    """写入 entry 时 enrichment（不影响 lookup key）。"""
    metadata = build_metadata_filters(rag_query, index_fingerprint=compute_index_fingerprint())
    return enrich_metadata_from_result(metadata, result)


def _chunk_from_dict(data: dict) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(data.get("chunk_id", "")),
        text=str(data.get("text", "")),
        company_name=str(data.get("company_name", "")),
        section_title=str(data.get("section_title", "")),
        page_start=int(data.get("page_start") or 0),
        page_end=int(data.get("page_end") or 0),
        display_name=str(data.get("display_name", "")),
        doc_id=str(data.get("doc_id", "")),
        source_pdf_path=str(data.get("source_pdf_path", "")),
        score=float(data.get("score") or 0.0),
        score_recall=float(data.get("score_recall") or 0.0),
        score_rerank=float(data["score_rerank"]) if data.get("score_rerank") is not None else None,
        score_vector=float(data["score_vector"]) if data.get("score_vector") is not None else None,
        score_bm25=float(data["score_bm25"]) if data.get("score_bm25") is not None else None,
        rank=int(data.get("rank") or 0),
        stage=str(data.get("stage") or "recall"),
    )


def _query_rewrite_from_dict(data: dict) -> QueryRewriteResult:
    return QueryRewriteResult(
        original_query=str(data.get("original_query", "")),
        query=str(data.get("query", "")),
        bm25_query=str(data.get("bm25_query", "")),
        stock_code=str(data.get("stock_code", "")),
        query_type=str(data.get("query_type", "factual")),
        compare_entities=list(data.get("compare_entities") or []),
        hybrid_vector_weight=float(data.get("hybrid_vector_weight") or 0.35),
        query_vector=None,
    )


def _citation_from_dict(data: dict) -> Citation:
    return Citation(
        index=int(data.get("index") or 0),
        chunk_id=str(data.get("chunk_id", "")),
        company_name=str(data.get("company_name", "")),
        section_title=str(data.get("section_title", "")),
        page_start=int(data.get("page_start") or 0),
        page_end=int(data.get("page_end") or 0),
        display_name=str(data.get("display_name", "")),
        score_rerank=float(data.get("score_rerank") or 0.0),
        doc_id=str(data.get("doc_id", "")),
        source_pdf_path=str(data.get("source_pdf_path", "")),
        filename=str(data.get("filename", "")),
    )


def _evidence_from_dict(data: dict) -> EvidenceCheckResult:
    return EvidenceCheckResult(
        passed=bool(data.get("passed", False)),
        top_rerank_score=float(data.get("top_rerank_score") or 0.0),
        refusal_reason=str(data.get("refusal_reason", "")),
        refusal_message=str(data.get("refusal_message", "")),
        refusal_detail=dict(data.get("refusal_detail") or {}),
        evidence_hits=list(data.get("evidence_hits") or []),
        citation_count=int(data.get("citation_count") or 0),
        checks=list(data.get("checks") or []),
    )


def search_result_from_payload(payload: dict) -> RAGSearchResult:
    rewrite = _query_rewrite_from_dict(payload["query_rewrite"])
    recall_data = payload["recall"]
    rerank_data = payload["rerank"]
    recall_hits = [_chunk_from_dict(item) for item in recall_data.get("hits") or []]
    rerank_hits = [_chunk_from_dict(item) for item in rerank_data.get("hits") or []]
    retrieve_result = HybridRetrieveResult(
        hits=[chunk.to_dict() for chunk in recall_hits],
        route=str(recall_data.get("route", "")),
        recall_top_k=int(recall_data.get("recall_top_k") or 0),
        query=str(payload.get("query", rewrite.query)),
        hit_count=int(recall_data.get("hit_count") or len(recall_hits)),
    )
    rerank_result = RerankStepResult(
        hits=[chunk.to_dict() for chunk in rerank_hits],
        query=str(payload.get("query", rewrite.query)),
        top_rerank_score=float(rerank_data.get("top_rerank_score") or 0.0),
        rerank_top_k=int(rerank_data.get("rerank_top_k") or len(rerank_hits)),
        hit_count=int(rerank_data.get("hit_count") or len(rerank_hits)),
    )
    cache_info = None
    if payload.get("cache"):
        cache_raw = payload["cache"]
        cache_info = CacheInfo(
            hit=bool(cache_raw.get("hit", False)),
            source=str(cache_raw.get("source", "pipeline")),
            similarity=cache_raw.get("similarity"),
            reason=str(cache_raw.get("reason", "")),
        )
    return RAGSearchResult(
        query=str(payload.get("query", rewrite.query)),
        query_rewrite=rewrite,
        retrieve_result=retrieve_result,
        rerank_result=rerank_result,
        recall_hits=recall_hits,
        rerank_hits=rerank_hits,
        top_rerank_score=float(payload.get("top_rerank_score") or rerank_result.top_rerank_score),
        cache=cache_info,
    )


def pipeline_result_from_payload(payload: dict) -> RAGPipelineResult:
    recall_hits = [_chunk_from_dict(item) for item in payload.get("recall_hits") or []]
    rerank_hits = [_chunk_from_dict(item) for item in payload.get("rerank_hits") or []]
    citations = [_citation_from_dict(item) for item in payload.get("citations") or []]
    rewrite = (
        _query_rewrite_from_dict(payload["query_rewrite"])
        if payload.get("query_rewrite")
        else None
    )
    evidence = (
        _evidence_from_dict(payload["evidence_check"])
        if payload.get("evidence_check")
        else None
    )
    retrieve_result = None
    if payload.get("retrieve_result"):
        rr = payload["retrieve_result"]
        retrieve_result = HybridRetrieveResult(
            hits=[chunk.to_dict() for chunk in recall_hits],
            route=str(rr.get("route", "")),
            recall_top_k=int(rr.get("recall_top_k") or 0),
            query=str(payload.get("query", "")),
            hit_count=int(rr.get("hit_count") or len(recall_hits)),
        )
    rerank_result = None
    if payload.get("rerank_result"):
        rr = payload["rerank_result"]
        rerank_result = RerankStepResult(
            hits=[chunk.to_dict() for chunk in rerank_hits],
            query=str(payload.get("query", "")),
            top_rerank_score=float(rr.get("top_rerank_score") or payload.get("top_rerank_score") or 0.0),
            rerank_top_k=int(rr.get("rerank_top_k") or len(rerank_hits)),
            hit_count=int(rr.get("hit_count") or len(rerank_hits)),
        )
    cache_info = None
    if payload.get("cache"):
        cache_raw = payload["cache"]
        cache_info = CacheInfo(
            hit=bool(cache_raw.get("hit", False)),
            source=str(cache_raw.get("source", "pipeline")),
            similarity=cache_raw.get("similarity"),
            reason=str(cache_raw.get("reason", "")),
        )
    return RAGPipelineResult(
        query=str(payload.get("query", "")),
        answer=str(payload.get("answer", "")),
        refused=bool(payload.get("refused", False)),
        refusal_reason=str(payload.get("refusal_reason", "")),
        top_rerank_score=float(payload.get("top_rerank_score") or 0.0),
        citations=citations,
        recall_hits=recall_hits,
        rerank_hits=rerank_hits,
        evidence_hits=list(payload.get("evidence_hits") or []),
        query_rewrite=rewrite,
        retrieve_result=retrieve_result,
        rerank_result=rerank_result,
        evidence_check=evidence,
        answer_generate=None,
        cache=cache_info,
    )


def _result_from_lookup(
    lookup: CacheLookupResult,
    *,
    scope: CacheScope,
    telemetry: CacheRequestTelemetry | None = None,
) -> RAGSearchResult | RAGPipelineResult:
    assert lookup.entry is not None
    payload = dict(lookup.entry.payload)
    if scope == CacheScope.SEARCH:
        result = search_result_from_payload(payload)
    else:
        result = pipeline_result_from_payload(payload)
    result.cache = CacheInfo.from_lookup(lookup, telemetry=telemetry)
    return result


def _lookup_with_trace(
    manager: CacheManager,
    rag_query: RAGQuery,
    config: RAGPipelineConfig,
    *,
    scope: CacheScope,
    rewrite: QueryRewriteResult | None = None,
) -> tuple[CacheLookupResult, list[CacheLookupAttempt]]:
    attempts: list[CacheLookupAttempt] = []
    ctx = build_cache_context(rag_query, config, scope=scope, rewrite=rewrite)
    timer = CacheRequestTimer()
    lookup = manager.lookup(ctx)
    layer = "l1_exact" if rewrite is None else "l2_semantic"
    attempts.append(attempt_from_lookup(lookup, latency_ms=timer.elapsed_ms(), layer=layer))
    return lookup, attempts


def _lookup_cached(
    pipeline: RAGPipeline,
    rag_query: RAGQuery,
    config: RAGPipelineConfig,
    *,
    scope: CacheScope,
) -> tuple[CacheLookupResult | None, QueryRewriteResult | None, list[CacheLookupAttempt], float]:
    """L1 →（可选 L2）查询，返回 (命中 lookup, rewrite, attempts, lookup_ms)。"""
    manager = get_cache_manager()
    attempts: list[CacheLookupAttempt] = []
    lookup_timer = CacheRequestTimer()

    lookup, l1_attempts = _lookup_with_trace(manager, rag_query, config, scope=scope)
    attempts.extend(l1_attempts)
    if lookup.hit:
        return lookup, None, attempts, lookup_timer.elapsed_ms()

    rewrite: QueryRewriteResult | None = None
    if manager.semantic_backend.implemented:
        rewrite = pipeline.query_rewrite(rag_query)
        lookup2, l2_attempts = _lookup_with_trace(
            manager, rag_query, config, scope=scope, rewrite=rewrite
        )
        attempts.extend(l2_attempts)
        if lookup2.hit:
            return lookup2, rewrite, attempts, lookup_timer.elapsed_ms()
        return None, rewrite, attempts, lookup_timer.elapsed_ms()

    return None, None, attempts, lookup_timer.elapsed_ms()


def lookup_search_cache(
    pipeline: RAGPipeline,
    rag_query: RAGQuery,
    config: RAGPipelineConfig,
    *,
    use_cache: bool = True,
) -> tuple[RAGSearchResult | None, QueryRewriteResult | None]:
    """L1 → L2 查询；命中返回 (result, rewrite)，未命中返回 (None, rewrite_or_none)。"""
    if not cache_enabled(use_cache=use_cache):
        return None, None

    lookup, rewrite, _attempts, _lookup_ms = _lookup_cached(
        pipeline, rag_query, config, scope=CacheScope.SEARCH
    )
    if lookup is not None and lookup.hit:
        return _result_from_lookup(lookup, scope=CacheScope.SEARCH), rewrite
    return None, rewrite


def run_search_with_cache(
    pipeline: RAGPipeline,
    rag_query: RAGQuery,
    config: RAGPipelineConfig,
    *,
    use_cache: bool = True,
) -> RAGSearchResult:
    """检索链路 + 缓存编排 + 遥测。"""
    manager = get_cache_manager()
    total_timer = CacheRequestTimer()
    telemetry = CacheRequestTelemetry(
        scope=CacheScope.SEARCH,
        stock_code=rag_query.stock_code,
        query_type=rag_query.query_type,
        cache_enabled=cache_enabled(use_cache=use_cache),
        cache_bypass=not use_cache,
    )

    if not cache_enabled(use_cache=use_cache):
        pipe_timer = CacheRequestTimer()
        result = pipeline._run_search_core(rag_query)
        telemetry = finalize_telemetry(
            telemetry,
            lookup=None,
            lookup_ms=0.0,
            pipeline_ms=pipe_timer.elapsed_ms(),
            total_ms=total_timer.elapsed_ms(),
        )
        telemetry.vector_retrieval = True
        telemetry.llm_called = False
        telemetry.reason = "cache_bypass" if not use_cache else "cache_disabled"
        emit_request_telemetry(manager, telemetry, query_preview=rag_query.query)
        result.cache = CacheInfo.from_telemetry(telemetry)
        return result

    lookup, rewrite, attempts, lookup_ms = _lookup_cached(
        pipeline, rag_query, config, scope=CacheScope.SEARCH
    )
    telemetry.attempts = attempts

    if lookup is not None and lookup.hit:
        telemetry = finalize_telemetry(
            telemetry,
            lookup=lookup,
            lookup_ms=lookup_ms,
            pipeline_ms=0.0,
            total_ms=total_timer.elapsed_ms(),
        )
        result = _result_from_lookup(lookup, scope=CacheScope.SEARCH, telemetry=telemetry)
        emit_request_telemetry(manager, telemetry, query_preview=rag_query.query)
        return result

    pipe_timer = CacheRequestTimer()
    result = pipeline._run_search_core(rag_query, rewrite=rewrite)
    pipeline_ms = pipe_timer.elapsed_ms()
    effective_rewrite = rewrite or result.query_rewrite
    if effective_rewrite is not None:
        store_search_cache(rag_query, config, result, rewrite=effective_rewrite, use_cache=True)

    last_lookup = lookup if lookup is not None else CacheLookupResult.miss("not_found")
    telemetry = finalize_telemetry(
        telemetry,
        lookup=last_lookup,
        lookup_ms=lookup_ms,
        pipeline_ms=pipeline_ms,
        total_ms=total_timer.elapsed_ms(),
    )
    if not telemetry.reason or telemetry.reason == "served":
        telemetry.reason = "not_found"
    telemetry.hit = False
    telemetry.source = "pipeline"
    telemetry.vector_retrieval = True
    telemetry.llm_called = False
    emit_request_telemetry(manager, telemetry, query_preview=rag_query.query)
    result.cache = CacheInfo.from_telemetry(telemetry)
    return result


def store_search_cache(
    rag_query: RAGQuery,
    config: RAGPipelineConfig,
    result: RAGSearchResult,
    *,
    rewrite: QueryRewriteResult,
    use_cache: bool = True,
) -> None:
    if not cache_enabled(use_cache=use_cache):
        return
    manager = get_cache_manager()
    ctx = build_cache_context(
        rag_query, config, scope=CacheScope.SEARCH, rewrite=rewrite
    )
    payload = result.to_dict()
    manager.store(
        ctx,
        payload=payload,
        refused=False,
        top_rerank_score=result.top_rerank_score,
        chunk_ids=extract_chunk_ids(payload),
        entry_metadata=entry_metadata_from_result(rag_query, result),
    )


def lookup_chat_cache(
    pipeline: RAGPipeline,
    rag_query: RAGQuery,
    config: RAGPipelineConfig,
    *,
    use_cache: bool = True,
) -> tuple[RAGPipelineResult | None, QueryRewriteResult | None]:
    if not cache_enabled(use_cache=use_cache):
        return None, None

    lookup, rewrite, _attempts, _lookup_ms = _lookup_cached(
        pipeline, rag_query, config, scope=CacheScope.CHAT
    )
    if lookup is not None and lookup.hit:
        return _result_from_lookup(lookup, scope=CacheScope.CHAT), rewrite
    return None, rewrite


def run_chat_with_cache(
    pipeline: RAGPipeline,
    rag_query: RAGQuery,
    config: RAGPipelineConfig,
    *,
    use_cache: bool = True,
) -> RAGPipelineResult:
    """完整 chat 链路 + 缓存编排 + 遥测。"""
    manager = get_cache_manager()
    total_timer = CacheRequestTimer()
    telemetry = CacheRequestTelemetry(
        scope=CacheScope.CHAT,
        stock_code=rag_query.stock_code,
        query_type=rag_query.query_type,
        cache_enabled=cache_enabled(use_cache=use_cache),
        cache_bypass=not use_cache,
    )

    if not cache_enabled(use_cache=use_cache):
        pipe_timer = CacheRequestTimer()
        result = pipeline._run_chat_core(rag_query, config)
        telemetry = finalize_telemetry(
            telemetry,
            lookup=None,
            lookup_ms=0.0,
            pipeline_ms=pipe_timer.elapsed_ms(),
            total_ms=total_timer.elapsed_ms(),
        )
        telemetry.vector_retrieval = True
        telemetry.llm_called = True
        telemetry.reason = "cache_bypass" if not use_cache else "cache_disabled"
        emit_request_telemetry(manager, telemetry, query_preview=rag_query.query)
        result.cache = CacheInfo.from_telemetry(telemetry)
        return result

    lookup, rewrite, attempts, lookup_ms = _lookup_cached(
        pipeline, rag_query, config, scope=CacheScope.CHAT
    )
    telemetry.attempts = attempts

    if lookup is not None and lookup.hit:
        telemetry = finalize_telemetry(
            telemetry,
            lookup=lookup,
            lookup_ms=lookup_ms,
            pipeline_ms=0.0,
            total_ms=total_timer.elapsed_ms(),
        )
        result = _result_from_lookup(lookup, scope=CacheScope.CHAT, telemetry=telemetry)
        emit_request_telemetry(manager, telemetry, query_preview=rag_query.query)
        return result

    pipe_timer = CacheRequestTimer()
    result = pipeline._run_chat_core(rag_query, config, rewrite=rewrite)
    pipeline_ms = pipe_timer.elapsed_ms()
    effective_rewrite = rewrite or result.query_rewrite
    if effective_rewrite is not None:
        store_chat_cache(rag_query, config, result, rewrite=effective_rewrite, use_cache=True)

    last_lookup = lookup if lookup is not None else CacheLookupResult.miss("not_found")
    telemetry = finalize_telemetry(
        telemetry,
        lookup=last_lookup,
        lookup_ms=lookup_ms,
        pipeline_ms=pipeline_ms,
        total_ms=total_timer.elapsed_ms(),
    )
    if not telemetry.reason or telemetry.reason == "served":
        telemetry.reason = "not_found"
    telemetry.hit = False
    telemetry.source = "pipeline"
    telemetry.vector_retrieval = True
    telemetry.llm_called = True
    emit_request_telemetry(manager, telemetry, query_preview=rag_query.query)
    result.cache = CacheInfo.from_telemetry(telemetry)
    return result


def store_chat_cache(
    rag_query: RAGQuery,
    config: RAGPipelineConfig,
    result: RAGPipelineResult,
    *,
    rewrite: QueryRewriteResult,
    use_cache: bool = True,
) -> None:
    if not cache_enabled(use_cache=use_cache):
        return
    manager = get_cache_manager()
    ctx = build_cache_context(
        rag_query, config, scope=CacheScope.CHAT, rewrite=rewrite
    )
    payload = result.to_dict()
    manager.store(
        ctx,
        payload=payload,
        refused=result.refused,
        top_rerank_score=result.top_rerank_score,
        chunk_ids=extract_chunk_ids(payload),
        entry_metadata=entry_metadata_from_result(rag_query, result),
    )
