"""Pipeline 结果 → API 响应的序列化（无检索 / 生成逻辑）。"""

from __future__ import annotations

from dataclasses import asdict

from api.schemas import (
    ChatResponse,
    CitationResponse,
    EvidenceCheckResponse,
    QueryRewriteResponse,
    RecallStageResponse,
    RerankStageResponse,
    RetrievedChunkResponse,
    SearchResponse,
)
from rag_types import Citation, QueryRewriteResult, RAGPipelineResult, RAGSearchResult, RetrievedChunk


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
    )


def _citation_to_response(citation: Citation) -> CitationResponse:
    return CitationResponse(**asdict(citation))


def chat_result_to_response(result: RAGPipelineResult) -> ChatResponse:
    evidence = result.evidence_check
    return ChatResponse(
        query=result.query,
        answer=result.answer,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        top_rerank_score=result.top_rerank_score,
        citations=[_citation_to_response(c) for c in result.citations],
        rerank_hits=[_chunk_to_response(chunk) for chunk in result.rerank_hits],
        evidence_check=EvidenceCheckResponse(
            passed=evidence.passed if evidence else not result.refused,
            top_rerank_score=evidence.top_rerank_score if evidence else result.top_rerank_score,
            refusal_reason=evidence.refusal_reason if evidence else result.refusal_reason,
        ),
    )
