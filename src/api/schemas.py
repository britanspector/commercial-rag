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


class CacheInfoResponse(BaseModel):
    hit: bool = False
    source: str = Field(default="pipeline", description="l1_exact | l2_semantic | pipeline | none")
    similarity: float | None = None
    reason: str = Field(default="", description="命中 served / 未命中原因")
    safety_ok: bool = True
    safety_reason: str = ""
    latency_ms: float = 0.0
    lookup_ms: float = 0.0
    pipeline_ms: float = 0.0
    vector_retrieval: bool = False
    llm_called: bool = False


class CacheStatsResponse(BaseModel):
    """进程内累计缓存统计（供监控 / 前端展示）。"""

    active: bool = False
    lookups: int = 0
    hits_l1: int = 0
    hits_l2: int = 0
    misses: int = 0
    lookup_hit_rate: float = 0.0
    requests: int = 0
    request_hits_l1: int = 0
    request_hits_l2: int = 0
    request_misses: int = 0
    l1_hit_rate: float = 0.0
    l2_hit_rate: float = 0.0
    total_hit_rate: float = 0.0
    avg_latency_ms: float = 0.0
    avg_hit_latency_ms: float = 0.0
    avg_miss_latency_ms: float = 0.0
    avg_latency_saved_ms: float = 0.0
    vector_retrievals_saved: int = 0
    llm_calls_saved: int = 0
    llm_call_reduction_rate: float = 0.0
    safety_rejects: int = 0
    stores: int = 0
    exact_entries: int = 0
    semantic_entries: int = 0
    backends: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    top_rerank_score: float
    query_rewrite: QueryRewriteResponse
    recall: RecallStageResponse
    rerank: RerankStageResponse
    cache: CacheInfoResponse = Field(default_factory=CacheInfoResponse)


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
    cache: CacheInfoResponse = Field(default_factory=CacheInfoResponse)


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    models_loaded: bool = False
    audit: dict[str, float | int | str | bool] = Field(
        default_factory=dict,
        description="审计库状态（enabled、backend、url_masked）",
    )
    cache: dict[str, float | int | str | bool] = Field(
        default_factory=dict,
        description="语义缓存累计统计摘要",
    )
    defaults: dict[str, float | int | str]


class UploadStageResponse(BaseModel):
    name: str
    status: str
    detail: str = ""


class JobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    progress: str = ""
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_ms: int | None = None
    result: dict = Field(default_factory=dict)
    error: str = ""


class EvalJobRequest(BaseModel):
    eval_type: Literal["generation", "retrieval", "ragas"] = Field(
        default="generation",
        description="generation=150题Pipeline；retrieval=检索评测；ragas=补跑RAGAS",
    )
    limit: int | None = Field(default=None, ge=1, le=500)
    skip_ragas: bool = Field(default=True, description="generation 时是否跳过内嵌 RAGAS")
    save_detail: bool = Field(default=True)
    resume: bool = Field(default=False)
    compare_routes: bool = Field(default=False, description="retrieval 时三路对比")
    route: Literal["vector", "bm25", "hybrid"] = "hybrid"
    top_k: int = Field(default=10, ge=1, le=50)
    legacy_retriever: bool = Field(default=False)
    pipeline_stage: Literal["recall", "rerank"] = "rerank"


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
    job_id: str = ""
    async_mode: bool = False
