"""
RAG 主链路功能模块：每步职责单一，输入输出结构化，便于调试 / 替换 / 评测。
"""

from pipeline.answer_generate import generate_answer
from pipeline.compose import compose_from_reranked_hits, compose_pipeline_result
from pipeline.evidence_check import check_evidence
from pipeline.hybrid_retrieve import hybrid_retrieve
from pipeline.query_rewrite import rewrite_query
from pipeline.rerank import rerank

__all__ = [
    "rewrite_query",
    "hybrid_retrieve",
    "rerank",
    "check_evidence",
    "generate_answer",
    "compose_pipeline_result",
    "compose_from_reranked_hits",
]
