"""
comparative 题型：分主体 Rerank + 配额合并，保证 Top-K 覆盖多家公司。
"""

from __future__ import annotations

import os

from rag_types import EntitySubQuery, RerankStepResult
from reranker import BGEReranker


def _comparative_entity_rerank_enabled() -> bool:
    raw = os.environ.get("RAG_COMPARATIVE_ENTITY_RERANK", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _company_matches_entity(company_name: str, entity: str) -> bool:
    company = (company_name or "").strip()
    ent = (entity or "").strip()
    if not company or not ent:
        return False
    return ent in company or company in ent


def _filter_hits_for_entity(hits: list[dict], entity: str) -> list[dict]:
    bucket = [h for h in hits if _company_matches_entity(str(h.get("company_name", "")), entity)]
    return bucket if bucket else list(hits)


def _sub_query_for_entity(
    entity: str,
    entity_sub_queries: list[EntitySubQuery],
    fallback_query: str,
) -> str:
    for sub in entity_sub_queries:
        if sub.entity.strip() == entity.strip():
            return sub.query
    return fallback_query


def distinct_companies_in_hits(hits: list[dict]) -> set[str]:
    """返回 hits 中 distinct company_name（用于测试）。"""
    names: set[str] = set()
    for hit in hits:
        name = str(hit.get("company_name", "")).strip()
        if name:
            names.add(name)
    return names


def rerank_comparative(
    query: str,
    hits: list[dict],
    reranker: BGEReranker,
    *,
    compare_entities: list[str],
    entity_sub_queries: list[EntitySubQuery],
    top_k: int = 5,
    normalize: bool = True,
) -> RerankStepResult:
    """
    对比题 Rerank：每主体用子查询打分并配额合并，避免 Top-K 被单公司占满。
    """
    entities = [e.strip() for e in compare_entities if e.strip()][:3]
    if not hits or len(entities) < 2 or not _comparative_entity_rerank_enabled():
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

    slots_per_entity = max(2, top_k // len(entities))
    merged: list[dict] = []
    seen: set[str] = set()

    for entity in entities:
        bucket = _filter_hits_for_entity(hits, entity)
        sub_q = _sub_query_for_entity(entity, entity_sub_queries, query)
        ranked = reranker.rerank_hits(
            sub_q,
            bucket,
            top_k=min(slots_per_entity, len(bucket)),
            normalize=normalize,
        )
        for hit in ranked:
            chunk_id = str(hit.get("chunk_id", ""))
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                merged.append(hit)

    if len(merged) < top_k:
        remaining = [h for h in hits if str(h.get("chunk_id", "")) not in seen]
        if remaining:
            fill = reranker.rerank_hits(
                query,
                remaining,
                top_k=top_k - len(merged),
                normalize=normalize,
            )
            for hit in fill:
                chunk_id = str(hit.get("chunk_id", ""))
                if chunk_id and chunk_id not in seen:
                    seen.add(chunk_id)
                    merged.append(hit)

    merged = merged[:top_k]
    top_score = float(
        merged[0].get("score_rerank") or merged[0].get("score") or 0.0
    ) if merged else 0.0

    return RerankStepResult(
        hits=merged,
        query=query,
        top_rerank_score=top_score,
        rerank_top_k=top_k,
        hit_count=len(merged),
    )
