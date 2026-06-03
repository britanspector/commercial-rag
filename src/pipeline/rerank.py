"""
rerank：对召回候选重打分并截断。

不改变 BGEReranker 算法，封装为独立步骤。
"""

from __future__ import annotations

from rag_types import RerankStepResult
from reranker import BGEReranker


def rerank(
    query: str,
    hits: list[dict],
    reranker: BGEReranker,
    *,
    top_k: int,
    normalize: bool = True,
) -> RerankStepResult:
    """
    输入：原始问题、召回 hits、Reranker
    输出：RerankStepResult（重排后 hits + top_rerank_score）
    """
    if not hits:
        return RerankStepResult(
            hits=[],
            query=query,
            top_rerank_score=0.0,
            rerank_top_k=top_k,
            hit_count=0,
        )

    reranked = reranker.rerank_hits(query, hits, top_k=top_k, normalize=normalize)
    top_score = float(
        reranked[0].get("score_rerank") or reranked[0].get("score") or 0.0
    ) if reranked else 0.0

    return RerankStepResult(
        hits=reranked,
        query=query,
        top_rerank_score=top_score,
        rerank_top_k=top_k,
        hit_count=len(reranked),
    )


def rerank_from_hits(
    query: str,
    hits: list[dict],
    *,
    top_k: int | None = None,
    normalize: bool = True,
) -> RerankStepResult:
    """
    离线评测专用：hits 已含 score_rerank 时直接封装，无需加载 Reranker。
    """
    effective_hits = hits[:top_k] if top_k is not None else hits
    top_score = float(
        effective_hits[0].get("score_rerank") or effective_hits[0].get("score") or 0.0
    ) if effective_hits else 0.0
    return RerankStepResult(
        hits=effective_hits,
        query=query,
        top_rerank_score=top_score,
        rerank_top_k=top_k or len(effective_hits),
        hit_count=len(effective_hits),
    )
