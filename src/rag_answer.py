"""
RAG 答案生成：向后兼容入口（逻辑已拆分至 pipeline/ 子模块）。
"""

from __future__ import annotations

import re

from pipeline.answer_generate import generate_answer
from pipeline.evidence_check import check_evidence
from pipeline.rerank import rerank_from_hits
from rag_constants import DEFAULT_RERANK_REFUSAL_THRESHOLD, REFUSAL_MESSAGE
from rag_tokens import must_tokens_match
from rag_types import Citation, RAGAnswer, build_citations

__all__ = [
    "Citation",
    "RAGAnswer",
    "build_citations",
    "generate_answer_with_citations",
    "is_answer_factually_supported",
]


def generate_answer_with_citations(
    query: str,
    hits: list[dict],
    refusal_threshold: float = DEFAULT_RERANK_REFUSAL_THRESHOLD,
) -> RAGAnswer:
    """向后兼容：内部走 evidence_check → answer_generate 两步。"""
    rerank_result = rerank_from_hits(query, hits)
    evidence = check_evidence(rerank_result, refusal_threshold=refusal_threshold)

    if not evidence.passed:
        return RAGAnswer(
            query=query,
            answer=REFUSAL_MESSAGE,
            refused=True,
            refusal_reason=evidence.refusal_reason,
            top_rerank_score=evidence.top_rerank_score,
            evidence_hits=evidence.evidence_hits,
        )

    generated = generate_answer(query, evidence, rerank_hits=hits)
    return RAGAnswer(
        query=query,
        answer=generated.answer,
        refused=False,
        top_rerank_score=generated.top_rerank_score,
        citations=generated.citations,
        evidence_hits=generated.evidence_hits,
    )


def is_answer_factually_supported(
    answer: str,
    must_contain_any: list[str],
    gold_answer: str = "",
) -> bool:
    if not answer or answer == REFUSAL_MESSAGE:
        return False
    if must_contain_any:
        return must_tokens_match(answer, must_contain_any)
    if gold_answer:
        gold_tokens = re.findall(r"[\u4e00-\u9fff]{2,}|\d+\.?\d*", gold_answer)
        if gold_tokens:
            matched = sum(1 for token in gold_tokens if token in answer)
            return matched >= min(2, len(gold_tokens))
    return len(answer) > 40
