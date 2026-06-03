"""
hybrid_retrieve：混合 / 向量 / BM25 召回。

不改变 HybridRetriever 算法，封装为独立步骤。
"""

from __future__ import annotations

from rag_types import HybridRetrieveResult, QueryRewriteResult
from retrieval import HybridRetriever, RecallRoute


def hybrid_retrieve(
    rewrite: QueryRewriteResult,
    retriever: HybridRetriever,
    *,
    route: RecallRoute,
    top_k: int,
) -> HybridRetrieveResult:
    """
    输入：QueryRewriteResult + HybridRetriever
    输出：HybridRetrieveResult（候选片段列表及召回元信息）
    """
    if rewrite.query_vector is None:
        raise ValueError("hybrid_retrieve 需要 query_vector，请先在 query_rewrite 中传入 embedder")

    hits = retriever.retrieve(
        route,
        rewrite.query,
        rewrite.query_vector,
        top_k,
        stock_code=rewrite.stock_code,
        query_type=rewrite.query_type,
    )
    return HybridRetrieveResult(
        hits=hits,
        route=route.value,
        recall_top_k=top_k,
        query=rewrite.query,
        hit_count=len(hits),
    )
