"""FastAPI 请求 / 响应模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")
    stock_code: str = Field(default="", description="股票代码（可选，用于召回增强）")
    query_type: Literal["factual", "comparative", "summary"] = Field(
        default="factual",
        description="问题类型",
    )
    recall_route: Literal["vector", "bm25", "hybrid"] = Field(
        default="hybrid",
        description="召回路线",
    )
    recall_top_k: int | None = Field(default=None, ge=1, le=200)
    rerank_top_k: int | None = Field(default=None, ge=1, le=50)
    refusal_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class QueryRewriteResponse(BaseModel):
    original_query: str
    query: str
    bm25_query: str
    stock_code: str
    query_type: str
    compare_entities: list[str]
    hybrid_vector_weight: float
    query_vector_dim: int | None = None


class RetrievedChunkResponse(BaseModel):
    rank: int
    chunk_id: str
    text: str
    company_name: str
    section_title: str
    page_start: int
    page_end: int
    display_name: str
    doc_id: str = ""
    source_pdf_path: str = ""
    score: float
    score_recall: float = 0.0
    score_rerank: float | None = None
    score_vector: float | None = None
    score_bm25: float | None = None


class RecallStageResponse(BaseModel):
    route: str
    recall_top_k: int
    hit_count: int
    hits: list[RetrievedChunkResponse]


class RerankStageResponse(BaseModel):
    rerank_top_k: int
    hit_count: int
    top_rerank_score: float
    hits: list[RetrievedChunkResponse]


class SearchResponse(BaseModel):
    query: str
    top_rerank_score: float
    query_rewrite: QueryRewriteResponse
    recall: RecallStageResponse
    rerank: RerankStageResponse


class CitationResponse(BaseModel):
    index: int
    chunk_id: str
    company_name: str
    section_title: str
    page_start: int
    page_end: int
    display_name: str
    score_rerank: float
    doc_id: str = ""
    source_pdf_path: str = ""
    filename: str = ""
    page_label: str = ""
    source_document: str = ""


class EvidenceCheckResponse(BaseModel):
    passed: bool
    top_rerank_score: float
    refusal_reason: str = ""
    refusal_message: str = ""
    citation_count: int = 0
    checks: list[dict[str, str | bool]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    query: str
    answer: str
    refused: bool
    refusal_reason: str
    refusal_message: str = ""
    top_rerank_score: float
    citations: list[CitationResponse]
    rerank_hits: list[RetrievedChunkResponse]
    evidence_check: EvidenceCheckResponse


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    models_loaded: bool = False
    audit: dict[str, float | int | str | bool] = Field(
        default_factory=dict,
        description="审计库状态（enabled、backend、url_masked）",
    )
    defaults: dict[str, float | int | str]


class UploadStageResponse(BaseModel):
    name: str
    status: str
    detail: str = ""


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    industry: str
    industry_label: str
    source_pdf_path: str
    display_name: str = ""
    company_name: str = ""
    stock_code: str = ""
    chunk_count: int
    retrievable_chunk_count: int
    milvus_rows_inserted: int
    milvus_total_rows: int
    bm25_total_chunks: int
    replaced_existing: bool
    stages: list[UploadStageResponse]
