"""
compose：将 evidence_check + answer_generate 组装为 RAGAnswer / RAGPipelineResult。
"""

from __future__ import annotations

from pipeline.answer_generate import generate_answer
from pipeline.evidence_check import check_evidence
from pipeline.rerank import rerank_from_hits
from rag_constants import DEFAULT_RERANK_REFUSAL_THRESHOLD, REFUSAL_MESSAGE
from rag_types import RAGAnswer
from rag_types import (
    AnswerGenerateResult,
    EvidenceCheckResult,
    HybridRetrieveResult,
    QueryRewriteResult,
    RAGPipelineResult,
    RerankStepResult,
)


def compose_pipeline_result(
    query: str,
    recall_hits: list[dict],
    rerank_result: RerankStepResult,
    *,
    refusal_threshold: float = DEFAULT_RERANK_REFUSAL_THRESHOLD,
    query_rewrite: QueryRewriteResult | None = None,
    retrieve_result: HybridRetrieveResult | None = None,
) -> RAGPipelineResult:
    """将 rerank 之后的步骤组装为完整 Pipeline 结果。"""
    evidence = check_evidence(rerank_result, refusal_threshold=refusal_threshold)
    answer_gen: AnswerGenerateResult | None = None

    if not evidence.passed:
        rag_answer = RAGAnswer(
            query=query,
            answer=REFUSAL_MESSAGE,
            refused=True,
            refusal_reason=evidence.refusal_reason,
            top_rerank_score=evidence.top_rerank_score,
            evidence_hits=evidence.evidence_hits,
        )
    else:
        answer_gen = generate_answer(query, evidence, rerank_hits=rerank_result.hits)
        rag_answer = RAGAnswer(
            query=query,
            answer=answer_gen.answer,
            refused=False,
            top_rerank_score=answer_gen.top_rerank_score,
            citations=answer_gen.citations,
            evidence_hits=answer_gen.evidence_hits,
        )

    return RAGPipelineResult.from_stages(
        query,
        recall_hits,
        rerank_result.hits,
        rag_answer,
        query_rewrite=query_rewrite,
        retrieve_result=retrieve_result,
        rerank_result=rerank_result,
        evidence_check=evidence,
        answer_generate=answer_gen,
    )


def compose_from_reranked_hits(
    query: str,
    recall_hits: list[dict],
    rerank_hits: list[dict],
    *,
    refusal_threshold: float = DEFAULT_RERANK_REFUSAL_THRESHOLD,
) -> RAGPipelineResult:
    """离线评测专用：已有 rerank 分数时跳过模型加载，直接走 evidence + generate。"""
    rerank_result = rerank_from_hits(query, rerank_hits)
    return compose_pipeline_result(
        query,
        recall_hits,
        rerank_result,
        refusal_threshold=refusal_threshold,
    )
