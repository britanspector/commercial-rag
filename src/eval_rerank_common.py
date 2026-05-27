"""
Rerank 评测共用逻辑（轻量，不导入 Milvus / Embedding）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eval_retrieval import EvalQuestion, is_hit_relevant, mrr, recall_at_k
from rag_answer import generate_answer_with_citations, is_answer_factually_supported

if TYPE_CHECKING:
    from reranker import BGEReranker

RECALL_POOL = 20
FINAL_TOP_K = 5


def top1_accuracy(question: EvalQuestion, hits: list[dict]) -> float:
    if not hits:
        return 0.0
    return 1.0 if is_hit_relevant(hits[0], question) else 0.0


def evaluate_strategy_hits(
    question: EvalQuestion,
    hits: list[dict],
    strategy: str,
) -> dict:
    relevant_ranks = [
        rank for rank, hit in enumerate(hits, start=1) if is_hit_relevant(hit, question)
    ]
    first_rank = min(relevant_ranks) if relevant_ranks else None
    retrieval_hit = first_rank is not None

    return {
        "question_id": question.id,
        "query_type": question.query_type,
        "strategy": strategy,
        "query": question.query,
        "gold_answer": question.gold_answer,
        "retrieval_hit": retrieval_hit,
        "first_relevant_rank": first_rank or "",
        "recall_at_5": recall_at_k(relevant_ranks, FINAL_TOP_K),
        "top1_accuracy": top1_accuracy(question, hits),
        "mrr": mrr(relevant_ranks),
        "top1_chunk_id": hits[0].get("chunk_id", "") if hits else "",
        "top1_score": hits[0].get("score_rerank", hits[0].get("score", "")) if hits else "",
    }


def _hits_have_rerank_scores(hits: list[dict]) -> bool:
    return bool(hits) and all("score_rerank" in hit for hit in hits)


def evaluate_answer_row(
    question: EvalQuestion,
    hits: list[dict],
    strategy: str,
    refusal_threshold: float,
    reranker: BGEReranker | None = None,
) -> dict:
    if _hits_have_rerank_scores(hits):
        scored_hits = hits
    else:
        if reranker is None:
            raise ValueError("hits 缺少 score_rerank，且未提供 reranker")
        scored_hits = reranker.rerank_hits(
            question.query, list(hits), top_k=len(hits), normalize=True
        )

    rag_answer = generate_answer_with_citations(
        question.query, scored_hits, refusal_threshold=refusal_threshold
    )
    retrieval_hit = any(is_hit_relevant(hit, question) for hit in scored_hits)
    fact_ok = is_answer_factually_supported(
        rag_answer.answer,
        question.must_contain_any,
        question.gold_answer,
    )

    if rag_answer.refused:
        refusal_ok = not retrieval_hit
    else:
        refusal_ok = retrieval_hit and fact_ok

    return {
        "question_id": question.id,
        "strategy": strategy,
        "refused": rag_answer.refused,
        "top_rerank_score": rag_answer.top_rerank_score,
        "answer_preview": rag_answer.answer[:200],
        "answer_factually_supported": fact_ok,
        "refusal_appropriate": refusal_ok,
        "retrieval_hit_in_top5": retrieval_hit,
    }


def aggregate_retrieval_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "question_count": n,
        "recall_at_5": sum(row["recall_at_5"] for row in rows) / n,
        "top1_accuracy": sum(row["top1_accuracy"] for row in rows) / n,
        "mrr": sum(row["mrr"] for row in rows) / n,
        "retrieval_hit_rate": sum(1 for row in rows if row["retrieval_hit"]) / n,
    }


def aggregate_answer_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "question_count": n,
        "answer_fact_accuracy": sum(row["answer_factually_supported"] for row in rows) / n,
        "refusal_appropriate_rate": sum(row["refusal_appropriate"] for row in rows) / n,
        "refusal_rate": sum(row["refused"] for row in rows) / n,
    }
