"""
RAG Pipeline 输入/输出数据结构（轻量，无模型依赖）。

各步骤 I/O 类型供 API、日志、评测复用；Pipeline 编排器在 rag_pipeline.py。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from reranker import hit_passage_text


@dataclass
class Citation:
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

    def source_document(self) -> str:
        return (
            self.display_name.strip()
            or self.filename.strip()
            or self.company_name.strip()
            or self.doc_id.strip()
        )

    def page_label(self) -> str:
        if not self.page_start:
            return "页码未知"
        if self.page_start == self.page_end or not self.page_end:
            return f"第{self.page_start}页"
        return f"第{self.page_start}-{self.page_end}页"

    def format_line(self) -> str:
        doc = self.source_document() or "未知文档"
        page = self.page_label()
        return (
            f"[{self.index}] {doc} — {self.section_title}，{page} "
            f"(chunk: {self.chunk_id}, doc: {self.doc_id or '-'}, rerank={self.score_rerank:.3f})"
        )


@dataclass
class RAGAnswer:
    query: str
    answer: str
    refused: bool
    refusal_reason: str = ""
    top_rerank_score: float = 0.0
    citations: list[Citation] = field(default_factory=list)
    evidence_hits: list[dict] = field(default_factory=list)


def build_citations(hits: list[dict]) -> list[Citation]:
    citations: list[Citation] = []
    for index, hit in enumerate(hits, start=1):
        citations.append(
            Citation(
                index=index,
                chunk_id=str(hit.get("chunk_id", "")),
                company_name=str(hit.get("company_name", "")),
                section_title=str(hit.get("section_title", "")),
                page_start=int(hit.get("page_start") or 0),
                page_end=int(hit.get("page_end") or 0),
                display_name=str(hit.get("display_name", "")),
                score_rerank=float(hit.get("score_rerank") or hit.get("score") or 0.0),
                doc_id=str(hit.get("doc_id", "")),
                source_pdf_path=str(hit.get("source_pdf_path", "")),
                filename=str(hit.get("filename", "")),
            )
        )
    return citations


@dataclass
class RAGQuery:
    """Pipeline 输入：用户问题及可选检索上下文。"""

    query: str
    stock_code: str = ""
    query_type: str = "factual"  # factual | comparative | summary

    @classmethod
    def from_text(cls, query: str) -> RAGQuery:
        return cls(query=query.strip())


@dataclass
class CacheInfo:
    """缓存命中元数据（随 Pipeline 结果返回）。"""

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

    def to_dict(self) -> dict[str, Any]:
        return {
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
        }

    @classmethod
    def from_lookup(cls, lookup: object, *, telemetry: object | None = None) -> CacheInfo:
        from cache.telemetry import CacheRequestTelemetry
        from cache.types import CacheLookupResult

        if isinstance(telemetry, CacheRequestTelemetry):
            return cls(
                hit=telemetry.hit,
                source=telemetry.source,
                similarity=telemetry.similarity,
                reason=telemetry.reason,
                safety_ok=telemetry.safety_ok,
                safety_reason=telemetry.safety_reason,
                latency_ms=telemetry.latency_ms,
                lookup_ms=telemetry.lookup_ms,
                pipeline_ms=telemetry.pipeline_ms,
                vector_retrieval=telemetry.vector_retrieval,
                llm_called=telemetry.llm_called,
            )

        if not isinstance(lookup, CacheLookupResult):
            raise TypeError("from_lookup 需要 CacheLookupResult 或 telemetry")
        if not lookup.hit:
            return cls(
                hit=False,
                source="pipeline",
                reason=lookup.reject_reason or "not_found",
                safety_ok=not bool(lookup.reject_reason),
                safety_reason=lookup.reject_reason or "",
            )
        layer = lookup.layer.value if lookup.layer else "pipeline"
        return cls(
            hit=True,
            source=layer,
            similarity=lookup.similarity,
            reason="served",
            vector_retrieval=False,
            llm_called=False,
        )

    @classmethod
    def from_telemetry(cls, telemetry: object) -> CacheInfo:
        from cache.telemetry import CacheRequestTelemetry

        if not isinstance(telemetry, CacheRequestTelemetry):
            raise TypeError("from_telemetry 需要 CacheRequestTelemetry")
        return cls(
            hit=telemetry.hit,
            source=telemetry.source,
            similarity=telemetry.similarity,
            reason=telemetry.reason,
            safety_ok=telemetry.safety_ok,
            safety_reason=telemetry.safety_reason,
            latency_ms=telemetry.latency_ms,
            lookup_ms=telemetry.lookup_ms,
            pipeline_ms=telemetry.pipeline_ms,
            vector_retrieval=telemetry.vector_retrieval,
            llm_called=telemetry.llm_called,
        )

    @classmethod
    def bypass(cls, *, reason: str = "cache_bypass", telemetry: object | None = None) -> CacheInfo:
        if telemetry is not None:
            return cls.from_telemetry(telemetry)
        return cls(hit=False, source="pipeline", reason=reason, vector_retrieval=True, llm_called=True)

    @classmethod
    def pipeline_miss(cls, *, reason: str = "not_found", telemetry: object | None = None) -> CacheInfo:
        if telemetry is not None:
            return cls.from_telemetry(telemetry)
        return cls(
            hit=False,
            source="pipeline",
            reason=reason,
            vector_retrieval=True,
            llm_called=True,
        )


@dataclass
class RetrievedChunk:
    """单条检索/重排片段的结构化表示。"""

    chunk_id: str
    text: str
    company_name: str
    section_title: str
    page_start: int
    page_end: int
    display_name: str
    doc_id: str = ""
    source_pdf_path: str = ""
    score: float = 0.0
    score_recall: float = 0.0
    score_rerank: float | None = None
    score_vector: float | None = None
    score_bm25: float | None = None
    rank: int = 0
    stage: str = "recall"  # recall | rerank

    @classmethod
    def from_hit(cls, hit: dict, *, rank: int, stage: str) -> RetrievedChunk:
        score = float(hit.get("score") or 0.0)
        score_rerank_raw = hit.get("score_rerank")
        score_rerank = float(score_rerank_raw) if score_rerank_raw is not None else None
        score_vector_raw = hit.get("score_vector")
        score_bm25_raw = hit.get("score_bm25")
        return cls(
            chunk_id=str(hit.get("chunk_id", "")),
            text=hit_passage_text(hit),
            company_name=str(hit.get("company_name", "")),
            section_title=str(hit.get("section_title", "")),
            page_start=int(hit.get("page_start") or 0),
            page_end=int(hit.get("page_end") or 0),
            display_name=str(hit.get("display_name", "")),
            doc_id=str(hit.get("doc_id", "")),
            source_pdf_path=str(hit.get("source_pdf_path", "")),
            score=score_rerank if stage == "rerank" and score_rerank is not None else score,
            score_recall=score if stage == "recall" else float(hit.get("score_recall") or score),
            score_rerank=score_rerank,
            score_vector=float(score_vector_raw) if score_vector_raw is not None else None,
            score_bm25=float(score_bm25_raw) if score_bm25_raw is not None else None,
            rank=rank,
            stage=stage,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EntitySubQuery:
    """对比题单主体检索子查询（含独立 embedding）。"""

    entity: str
    query: str
    query_vector: list[float] | None = None


@dataclass
class QueryRewriteResult:
    """query_rewrite 步骤输出。"""

    original_query: str
    query: str
    bm25_query: str
    stock_code: str = ""
    query_type: str = "factual"
    compare_entities: list[str] = field(default_factory=list)
    hybrid_vector_weight: float = 0.35
    query_vector: list[float] | None = None
    entity_sub_queries: list[EntitySubQuery] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.query_vector is not None:
            payload["query_vector_dim"] = len(self.query_vector)
            payload["query_vector"] = None
        if self.entity_sub_queries:
            payload["entity_sub_queries"] = [
                {
                    "entity": sub.entity,
                    "query": sub.query,
                    "query_vector_dim": len(sub.query_vector) if sub.query_vector else None,
                }
                for sub in self.entity_sub_queries
            ]
        return payload


@dataclass
class HybridRetrieveResult:
    """hybrid_retrieve 步骤输出。"""

    hits: list[dict]
    route: str
    recall_top_k: int
    query: str
    hit_count: int = 0

    def __post_init__(self) -> None:
        if not self.hit_count:
            self.hit_count = len(self.hits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "recall_top_k": self.recall_top_k,
            "query": self.query,
            "hit_count": self.hit_count,
            "hits": self.hits,
        }


@dataclass
class RerankStepResult:
    """rerank 步骤输出。"""

    hits: list[dict]
    query: str
    top_rerank_score: float
    rerank_top_k: int
    hit_count: int = 0

    def __post_init__(self) -> None:
        if not self.hit_count:
            self.hit_count = len(self.hits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "top_rerank_score": self.top_rerank_score,
            "rerank_top_k": self.rerank_top_k,
            "hit_count": self.hit_count,
            "hits": self.hits,
        }


@dataclass
class EvidenceCheckResult:
    """evidence_check 步骤输出：判断证据是否足以生成答案。"""

    passed: bool
    top_rerank_score: float
    refusal_reason: str = ""
    refusal_message: str = ""
    refusal_detail: dict[str, Any] = field(default_factory=dict)
    evidence_hits: list[dict] = field(default_factory=list)
    citation_count: int = 0
    checks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "top_rerank_score": self.top_rerank_score,
            "refusal_reason": self.refusal_reason,
            "refusal_message": self.refusal_message,
            "refusal_detail": self.refusal_detail,
            "citation_count": self.citation_count,
            "checks": self.checks,
            "evidence_hit_count": len(self.evidence_hits),
        }


@dataclass
class AnswerGenerateResult:
    """answer_generate 步骤输出。"""

    query: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    evidence_hits: list[dict] = field(default_factory=list)
    top_rerank_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "top_rerank_score": self.top_rerank_score,
            "citations": [asdict(c) for c in self.citations],
            "evidence_hits": self.evidence_hits,
        }


@dataclass
class RAGPipelineResult:
    """Pipeline 完整输出：答案 + 各阶段检索片段 + 引用 + 拒答信息。"""

    query: str
    answer: str
    refused: bool
    refusal_reason: str = ""
    top_rerank_score: float = 0.0
    citations: list[Citation] = field(default_factory=list)
    recall_hits: list[RetrievedChunk] = field(default_factory=list)
    rerank_hits: list[RetrievedChunk] = field(default_factory=list)
    evidence_hits: list[dict] = field(default_factory=list)
    query_rewrite: QueryRewriteResult | None = None
    retrieve_result: HybridRetrieveResult | None = None
    rerank_result: RerankStepResult | None = None
    evidence_check: EvidenceCheckResult | None = None
    answer_generate: AnswerGenerateResult | None = None
    cache: CacheInfo | None = None

    @classmethod
    def from_stages(
        cls,
        query: str,
        recall_hits: list[dict],
        rerank_hits: list[dict],
        rag_answer: RAGAnswer,
        *,
        query_rewrite: QueryRewriteResult | None = None,
        retrieve_result: HybridRetrieveResult | None = None,
        rerank_result: RerankStepResult | None = None,
        evidence_check: EvidenceCheckResult | None = None,
        answer_generate: AnswerGenerateResult | None = None,
    ) -> RAGPipelineResult:
        recall_with_rerank_score = []
        rerank_by_id = {hit.get("chunk_id"): hit for hit in rerank_hits}
        for hit in recall_hits:
            enriched = dict(hit)
            reranked = rerank_by_id.get(hit.get("chunk_id"))
            if reranked is not None:
                enriched["score_rerank"] = reranked.get("score_rerank", reranked.get("score"))
            recall_with_rerank_score.append(enriched)

        return cls(
            query=query,
            answer=rag_answer.answer,
            refused=rag_answer.refused,
            refusal_reason=rag_answer.refusal_reason,
            top_rerank_score=rag_answer.top_rerank_score,
            citations=list(rag_answer.citations),
            recall_hits=[
                RetrievedChunk.from_hit(hit, rank=index, stage="recall")
                for index, hit in enumerate(recall_hits, start=1)
            ],
            rerank_hits=[
                RetrievedChunk.from_hit(hit, rank=index, stage="rerank")
                for index, hit in enumerate(rerank_hits, start=1)
            ],
            evidence_hits=recall_with_rerank_score[:3] if rag_answer.refused else list(rag_answer.evidence_hits),
            query_rewrite=query_rewrite,
            retrieve_result=retrieve_result,
            rerank_result=rerank_result,
            evidence_check=evidence_check,
            answer_generate=answer_generate,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": self.query,
            "answer": self.answer,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "top_rerank_score": self.top_rerank_score,
            "citations": [asdict(c) for c in self.citations],
            "recall_hits": [chunk.to_dict() for chunk in self.recall_hits],
            "rerank_hits": [chunk.to_dict() for chunk in self.rerank_hits],
            "evidence_hits": self.evidence_hits,
        }
        if self.query_rewrite is not None:
            payload["query_rewrite"] = self.query_rewrite.to_dict()
        if self.retrieve_result is not None:
            payload["retrieve_result"] = {
                "route": self.retrieve_result.route,
                "recall_top_k": self.retrieve_result.recall_top_k,
                "hit_count": self.retrieve_result.hit_count,
            }
        if self.rerank_result is not None:
            payload["rerank_result"] = {
                "top_rerank_score": self.rerank_result.top_rerank_score,
                "rerank_top_k": self.rerank_result.rerank_top_k,
                "hit_count": self.rerank_result.hit_count,
            }
        if self.evidence_check is not None:
            payload["evidence_check"] = self.evidence_check.to_dict()
        if self.answer_generate is not None:
            payload["answer_generate"] = {
                "top_rerank_score": self.answer_generate.top_rerank_score,
                "citation_count": len(self.answer_generate.citations),
            }
        if self.cache is not None:
            payload["cache"] = self.cache.to_dict()
        return payload


@dataclass
class RAGSearchResult:
    """Pipeline 检索链路输出（query_rewrite → hybrid_retrieve → rerank，不含生成）。"""

    query: str
    query_rewrite: QueryRewriteResult
    retrieve_result: HybridRetrieveResult
    rerank_result: RerankStepResult
    recall_hits: list[RetrievedChunk] = field(default_factory=list)
    rerank_hits: list[RetrievedChunk] = field(default_factory=list)
    top_rerank_score: float = 0.0
    cache: CacheInfo | None = None

    @classmethod
    def from_steps(
        cls,
        rewrite: QueryRewriteResult,
        retrieved: HybridRetrieveResult,
        reranked: RerankStepResult,
    ) -> RAGSearchResult:
        return cls(
            query=rewrite.query,
            query_rewrite=rewrite,
            retrieve_result=retrieved,
            rerank_result=reranked,
            recall_hits=[
                RetrievedChunk.from_hit(hit, rank=index, stage="recall")
                for index, hit in enumerate(retrieved.hits, start=1)
            ],
            rerank_hits=[
                RetrievedChunk.from_hit(hit, rank=index, stage="rerank")
                for index, hit in enumerate(reranked.hits, start=1)
            ],
            top_rerank_score=reranked.top_rerank_score,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": self.query,
            "top_rerank_score": self.top_rerank_score,
            "query_rewrite": self.query_rewrite.to_dict(),
            "recall": {
                "route": self.retrieve_result.route,
                "recall_top_k": self.retrieve_result.recall_top_k,
                "hit_count": self.retrieve_result.hit_count,
                "hits": [chunk.to_dict() for chunk in self.recall_hits],
            },
            "rerank": {
                "rerank_top_k": self.rerank_result.rerank_top_k,
                "hit_count": self.rerank_result.hit_count,
                "top_rerank_score": self.rerank_result.top_rerank_score,
                "hits": [chunk.to_dict() for chunk in self.rerank_hits],
            },
        }
        if self.cache is not None:
            payload["cache"] = self.cache.to_dict()
        return payload
