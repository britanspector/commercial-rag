"""
evidence_check：判断重排后证据是否足以生成答案。

当前规则（与原有 generate_answer_with_citations 一致）：
- 无 hits → no_hits
- Top-1 rerank 分低于阈值 → low_rerank_score
"""

from __future__ import annotations

from rag_constants import DEFAULT_RERANK_REFUSAL_THRESHOLD
from rag_types import EvidenceCheckResult, RerankStepResult


def check_evidence(
    rerank_result: RerankStepResult | list[dict],
    *,
    refusal_threshold: float = DEFAULT_RERANK_REFUSAL_THRESHOLD,
) -> EvidenceCheckResult:
    """
    输入：RerankStepResult 或 rerank 后的 hits 列表
    输出：EvidenceCheckResult（是否通过、拒答原因、可用证据片段）
    """
    if isinstance(rerank_result, RerankStepResult):
        hits = rerank_result.hits
        top_score = rerank_result.top_rerank_score
    else:
        hits = rerank_result
        top_score = float(
            hits[0].get("score_rerank") or hits[0].get("score") or 0.0
        ) if hits else 0.0

    if not hits:
        return EvidenceCheckResult(
            passed=False,
            top_rerank_score=0.0,
            refusal_reason="no_hits",
            evidence_hits=[],
        )

    if top_score < refusal_threshold:
        return EvidenceCheckResult(
            passed=False,
            top_rerank_score=top_score,
            refusal_reason="low_rerank_score",
            evidence_hits=hits[:3],
        )

    return EvidenceCheckResult(
        passed=True,
        top_rerank_score=top_score,
        refusal_reason="",
        evidence_hits=hits[:3],
    )
