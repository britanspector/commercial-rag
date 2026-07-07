"""
query_rewrite：查询改写与编码（BM25 扩展、对比实体、混合权重、向量编码）。

不改变底层增强逻辑，封装 query_enhance + encode_query 为独立可测步骤。
"""

from __future__ import annotations

from typing import Any

from eval_retrieval import encode_query
from query_enhance import (
    build_comparative_sub_queries,
    enhance_bm25_query,
    extract_compare_entities,
    hybrid_vector_weight,
)
from rag_types import EntitySubQuery, RAGQuery, QueryRewriteResult
from retrieval import DEFAULT_HYBRID_VECTOR_WEIGHT


def rewrite_query(
    rag_query: RAGQuery,
    *,
    embedder: Any | None = None,
    hybrid_vector_weight_default: float = DEFAULT_HYBRID_VECTOR_WEIGHT,
) -> QueryRewriteResult:
    """
    将用户问题转为检索可用的结构化查询。

    输入：RAGQuery（原始问题 + 可选 stock_code / query_type）
    输出：QueryRewriteResult（BM25 扩展词、对比实体、混合权重、可选 query_vector）
    """
    query = rag_query.query.strip()
    bm25_query = enhance_bm25_query(query, rag_query.stock_code)
    compare_entities: list[str] = []
    if rag_query.query_type == "comparative":
        compare_entities = extract_compare_entities(query)
    vector_weight = hybrid_vector_weight(
        rag_query.query_type, query, default=hybrid_vector_weight_default
    )
    query_vector = encode_query(embedder, query) if embedder is not None else None

    entity_sub_queries: list[EntitySubQuery] = []
    if rag_query.query_type == "comparative" and len(compare_entities) >= 2:
        for entity, sub_q in build_comparative_sub_queries(query, compare_entities):
            sub_vector = encode_query(embedder, sub_q) if embedder is not None else None
            entity_sub_queries.append(
                EntitySubQuery(entity=entity, query=sub_q, query_vector=sub_vector)
            )

    return QueryRewriteResult(
        original_query=query,
        query=query,
        bm25_query=bm25_query,
        stock_code=rag_query.stock_code,
        query_type=rag_query.query_type,
        compare_entities=compare_entities,
        hybrid_vector_weight=vector_weight,
        query_vector=query_vector,
        entity_sub_queries=entity_sub_queries,
    )
