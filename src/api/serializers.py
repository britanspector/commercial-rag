"""Pipeline 结果 → API 响应的序列化（无检索 / 生成逻辑）。"""

from __future__ import annotations

from dataclasses import asdict

from api.schemas import (
    CacheInfoResponse,
    ChatResponse,
    CitationResponse,
    EvidenceCheckResponse,
    QueryRewriteResponse,
    RecallStageResponse,
    RerankStageResponse,
    RetrievedChunkResponse,
    SearchResponse,
    UploadResponse,
    UploadStageResponse,
)
from rag_types import Citation, QueryRewriteResult, RAGPipelineResult, RAGSearchResult, RetrievedChunk


def _cache_to_response(cache) -> CacheInfoResponse:
    if cache is None:
        return CacheInfoResponse()
    return CacheInfoResponse(
        hit=cache.hit,
        source=cache.source,
        similarity=cache.similarity,
        reason=cache.reason,
        safety_ok=cache.safety_ok,
        safety_reason=cache.safety_reason,
        latency_ms=cache.latency_ms,
        lookup_ms=cache.lookup_ms,
        pipeline_ms=cache.pipeline_ms,
        vector_retrieval=cache.vector_retrieval,
        llm_called=cache.llm_called,
    )


def ingest_result_to_response(result) -> UploadResponse:
    from pipeline.ingest import IngestResult

    if not isinstance(result, IngestResult):
        raise TypeError("ingest_result_to_response 需要 IngestResult")

    return UploadResponse(
        doc_id=result.doc_id,
        filename=result.filename,
        industry=result.industry,
        industry_label=result.industry_label,
        source_pdf_path=result.source_pdf_path,
        display_name=result.display_name,
        company_name=result.company_name,
        stock_code=result.stock_code,
        chunk_count=result.chunk_count,
        retrievable_chunk_count=result.retrievable_chunk_count,
        milvus_rows_inserted=result.milvus_rows_inserted,
        milvus_total_rows=result.milvus_total_rows,
        bm25_total_chunks=result.bm25_total_chunks,
        replaced_existing=result.replaced_existing,
        stages=[
            UploadStageResponse(name=stage.name, status=stage.status, detail=stage.detail)
            for stage in result.stages
        ],
    )


def _chunk_to_response(chunk: RetrievedChunk) -> RetrievedChunkResponse:
    return RetrievedChunkResponse(**chunk.to_dict())


def _rewrite_to_response(rewrite: QueryRewriteResult) -> QueryRewriteResponse:
    payload = rewrite.to_dict()
    return QueryRewriteResponse(
        original_query=payload["original_query"],
        query=payload["query"],
        bm25_query=payload["bm25_query"],
        stock_code=payload["stock_code"],
        query_type=payload["query_type"],
        compare_entities=payload["compare_entities"],
        hybrid_vector_weight=payload["hybrid_vector_weight"],
        query_vector_dim=payload.get("query_vector_dim"),
    )


def search_result_to_response(result: RAGSearchResult) -> SearchResponse:
    data = result.to_dict()
    return SearchResponse(
        query=data["query"],
        top_rerank_score=data["top_rerank_score"],
        query_rewrite=_rewrite_to_response(result.query_rewrite),
        recall=RecallStageResponse(
            route=data["recall"]["route"],
            recall_top_k=data["recall"]["recall_top_k"],
            hit_count=data["recall"]["hit_count"],
            hits=[RetrievedChunkResponse(**hit) for hit in data["recall"]["hits"]],
        ),
        rerank=RerankStageResponse(
            rerank_top_k=data["rerank"]["rerank_top_k"],
            hit_count=data["rerank"]["hit_count"],
            top_rerank_score=data["rerank"]["top_rerank_score"],
            hits=[RetrievedChunkResponse(**hit) for hit in data["rerank"]["hits"]],
        ),
        cache=_cache_to_response(result.cache),
    )


def _citation_to_response(citation: Citation) -> CitationResponse:
    payload = asdict(citation)
    payload["page_label"] = citation.page_label()
    payload["source_document"] = citation.source_document()
    return CitationResponse(**payload)


def chat_result_to_response(result: RAGPipelineResult) -> ChatResponse:
    evidence = result.evidence_check
    refusal_message = evidence.refusal_message if evidence else ""
    if result.refused and not refusal_message:
        refusal_message = result.answer
    return ChatResponse(
        query=result.query,
        answer=result.answer,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        refusal_message=refusal_message,
        top_rerank_score=result.top_rerank_score,
        citations=[_citation_to_response(c) for c in result.citations],
        rerank_hits=[_chunk_to_response(chunk) for chunk in result.rerank_hits],
        evidence_check=EvidenceCheckResponse(
            passed=evidence.passed if evidence else not result.refused,
            top_rerank_score=evidence.top_rerank_score if evidence else result.top_rerank_score,
            refusal_reason=evidence.refusal_reason if evidence else result.refusal_reason,
            refusal_message=refusal_message,
            citation_count=evidence.citation_count if evidence else len(result.citations),
            checks=evidence.checks if evidence else [],
        ),
        cache=_cache_to_response(result.cache),
    )
