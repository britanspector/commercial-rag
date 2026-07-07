"""
RAG 统一流水线：query_rewrite → hybrid_retrieve → rerank → evidence_check → answer_generate。

FastAPI、CLI、离线评测应通过本模块调用同一条主链路；各步骤实现位于 pipeline/ 子包。
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from hf_env import bootstrap_hf_cache

bootstrap_hf_cache()

from bm25_store import DEFAULT_INDEX_PATH
from embed_chunks import EMBED_DIM, OUTPUT_MILVUS_DB, load_embedder, resolve_device
from pipeline.answer_generate import generate_answer
from pipeline.compose import compose_from_reranked_hits, compose_pipeline_result
from pipeline.evidence_check import check_evidence
from pipeline.comparative_rerank import rerank_comparative
from pipeline.hybrid_retrieve import hybrid_retrieve
from pipeline.query_rewrite import rewrite_query
from pipeline.rerank import rerank
from rag_types import (
    AnswerGenerateResult,
    CacheInfo,
    EvidenceCheckResult,
    HybridRetrieveResult,
    QueryRewriteResult,
    RAGAnswer,
    RAGPipelineResult,
    RAGQuery,
    RAGSearchResult,
    RerankStepResult,
)
from rag_answer import generate_answer_with_citations
from rag_constants import (
    DEFAULT_RERANK_REFUSAL_THRESHOLD,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_RECALL_TOP_K,
    REFUSAL_MESSAGE,
)
from reranker import BGEReranker
from retrieval import (
    DEFAULT_HYBRID_POOL_SIZE,
    DEFAULT_HYBRID_VECTOR_WEIGHT,
    DEFAULT_OUTPUT_FIELDS,
    HybridRetriever,
    RecallRoute,
    _hits_from_vector,
)

__all__ = [
    "RAGPipeline",
    "RAGPipelineConfig",
    "RAGQuery",
    "RAGPipelineResult",
    "RAGSearchResult",
    "CacheInfo",
    "RAGAnswer",
    "QueryRewriteResult",
    "HybridRetrieveResult",
    "RerankStepResult",
    "EvidenceCheckResult",
    "AnswerGenerateResult",
    "generate_answer_with_citations",
    "REFUSAL_MESSAGE",
    "DEFAULT_RERANK_REFUSAL_THRESHOLD",
]


@dataclass
class RAGPipelineConfig:
    """Pipeline 可调参数。"""

    recall_top_k: int = DEFAULT_RECALL_TOP_K
    rerank_top_k: int = DEFAULT_RERANK_TOP_K
    refusal_threshold: float = DEFAULT_RERANK_REFUSAL_THRESHOLD
    recall_route: RecallRoute = RecallRoute.HYBRID
    hybrid_vector_weight: float = DEFAULT_HYBRID_VECTOR_WEIGHT
    hybrid_pool_size: int = DEFAULT_HYBRID_POOL_SIZE


class RAGPipeline:
    """
    统一 RAG 主链路编排器，按步骤委托 pipeline/ 子模块。

    典型用法::

        pipeline = RAGPipeline()
        result = pipeline.run("京仪装备2026E毛利率预测是多少？")

    分步调用（便于日志 / 单步调试 / 替换）::

        rewrite = pipeline.query_rewrite(query)
        retrieved = pipeline.hybrid_retrieve(rewrite)
        reranked = pipeline.rerank_step(rewrite.query, retrieved.hits)
        evidence = pipeline.evidence_check(reranked)
        answer = pipeline.answer_generate(rewrite.query, evidence, reranked.hits)
    """

    def __init__(
        self,
        config: RAGPipelineConfig | None = None,
        *,
        recall_top_k: int | None = None,
        rerank_top_k: int | None = None,
        refusal_threshold: float | None = None,
        recall_route: RecallRoute | str | None = None,
        hybrid_vector_weight: float | None = None,
        hybrid_pool_size: int | None = None,
    ) -> None:
        self.config = config or RAGPipelineConfig()
        if recall_top_k is not None:
            self.config.recall_top_k = recall_top_k
        if rerank_top_k is not None:
            self.config.rerank_top_k = rerank_top_k
        if refusal_threshold is not None:
            self.config.refusal_threshold = refusal_threshold
        if recall_route is not None:
            self.config.recall_route = (
                recall_route if isinstance(recall_route, RecallRoute) else RecallRoute(recall_route)
            )
        if hybrid_vector_weight is not None:
            self.config.hybrid_vector_weight = hybrid_vector_weight
        if hybrid_pool_size is not None:
            self.config.hybrid_pool_size = hybrid_pool_size

        self._embedder = None
        self._retriever: HybridRetriever | None = None
        self._reranker: BGEReranker | None = None
        self._config_lock = threading.Lock()

    @property
    def recall_top_k(self) -> int:
        return self.config.recall_top_k

    @property
    def rerank_top_k(self) -> int:
        return self.config.rerank_top_k

    @property
    def refusal_threshold(self) -> float:
        return self.config.refusal_threshold

    @property
    def is_loaded(self) -> bool:
        return self._embedder is not None

    def _ensure_loaded(self) -> None:
        if self._embedder is not None:
            return
        device = resolve_device()
        self._embedder = load_embedder(device)
        self._retriever = HybridRetriever.from_paths(
            OUTPUT_MILVUS_DB,
            vector_dim=EMBED_DIM,
            bm25_index_path=DEFAULT_INDEX_PATH,
            hybrid_vector_weight=self.config.hybrid_vector_weight,
            hybrid_pool_size=self.config.hybrid_pool_size,
            output_fields=DEFAULT_OUTPUT_FIELDS,
        )
        self._retriever.milvus_store.load()
        self._reranker = BGEReranker(device=device)

    def close(self) -> None:
        if self._retriever is not None:
            self._retriever.close()
        self._retriever = None
        self._embedder = None
        self._reranker = None

    def _normalize_query(self, query: RAGQuery | str) -> RAGQuery:
        if isinstance(query, RAGQuery):
            return query
        return RAGQuery.from_text(str(query))

    # --- 分步接口 ---

    def query_rewrite(self, query: RAGQuery | str) -> QueryRewriteResult:
        """步骤 1：查询改写与向量编码。"""
        self._ensure_loaded()
        rag_query = self._normalize_query(query)
        return rewrite_query(
            rag_query,
            embedder=self._embedder,
            hybrid_vector_weight_default=self.config.hybrid_vector_weight,
        )

    def hybrid_retrieve(self, rewrite: QueryRewriteResult) -> HybridRetrieveResult:
        """步骤 2：混合 / 向量 / BM25 召回。"""
        self._ensure_loaded()
        assert self._retriever is not None
        return hybrid_retrieve(
            rewrite,
            self._retriever,
            route=self.config.recall_route,
            top_k=self.config.recall_top_k,
        )

    def rerank_step(self, query: str, hits: list[dict]) -> RerankStepResult:
        """步骤 3：Rerank 重排。"""
        self._ensure_loaded()
        assert self._reranker is not None
        return rerank(query, hits, self._reranker, top_k=self.config.rerank_top_k)

    def rerank_step_for_rewrite(
        self,
        rewrite: QueryRewriteResult,
        hits: list[dict],
    ) -> RerankStepResult:
        """对比题走分主体 Rerank + 配额合并；其余题型与 rerank_step 相同。"""
        self._ensure_loaded()
        assert self._reranker is not None
        if (
            rewrite.query_type == "comparative"
            and len(rewrite.compare_entities) >= 2
            and rewrite.entity_sub_queries
        ):
            return rerank_comparative(
                rewrite.query,
                hits,
                self._reranker,
                compare_entities=rewrite.compare_entities,
                entity_sub_queries=rewrite.entity_sub_queries,
                top_k=self.config.rerank_top_k,
            )
        return rerank(rewrite.query, hits, self._reranker, top_k=self.config.rerank_top_k)

    def evidence_check(
        self,
        rerank_result: RerankStepResult,
        *,
        query: str = "",
        stock_code: str = "",
        query_type: str = "factual",
        compare_entities: list[str] | None = None,
    ) -> EvidenceCheckResult:
        """步骤 4：证据充分性校验。"""
        return check_evidence(
            rerank_result,
            refusal_threshold=self.config.refusal_threshold,
            query=query,
            stock_code=stock_code,
            query_type=query_type,
            compare_entities=compare_entities,
        )

    def answer_generate(
        self,
        query: str,
        evidence: EvidenceCheckResult,
        rerank_hits: list[dict],
        *,
        query_type: str = "factual",
        compare_entities: list[str] | None = None,
    ) -> AnswerGenerateResult:
        """步骤 5：带引用答案生成（须 evidence.passed）。"""
        return generate_answer(
            query,
            evidence,
            rerank_hits=rerank_hits,
            query_type=query_type,
            compare_entities=compare_entities,
        )

    # --- 兼容 / 组合接口 ---

    def retrieve(self, query: RAGQuery | str) -> list[dict]:
        """召回阶段（兼容旧接口）。"""
        rewrite = self.query_rewrite(query)
        return self.hybrid_retrieve(rewrite).hits

    def retrieve_vector(self, query: str, top_k: int | None = None) -> list[dict]:
        """纯向量召回（兼容旧接口）。"""
        self._ensure_loaded()
        assert self._embedder is not None and self._retriever is not None
        from eval_retrieval import encode_query

        query_vector = encode_query(self._embedder, query)
        return _hits_from_vector(
            self._retriever.milvus_store,
            query_vector,
            top_k or self.config.recall_top_k,
            DEFAULT_OUTPUT_FIELDS,
        )

    def rerank(self, query: str, hits: list[dict]) -> list[dict]:
        """重排阶段（兼容旧接口，返回 hits 列表）。"""
        return self.rerank_step(query, hits).hits

    def generate(
        self,
        query: str,
        recall_hits: list[dict],
        rerank_hits: list[dict],
        *,
        refusal_threshold: float | None = None,
    ) -> RAGPipelineResult:
        """生成阶段（兼容旧接口）。"""
        from pipeline.rerank import rerank_from_hits

        threshold = (
            self.config.refusal_threshold if refusal_threshold is None else refusal_threshold
        )
        rerank_result = rerank_from_hits(query, rerank_hits, top_k=len(rerank_hits))
        return compose_pipeline_result(
            query,
            recall_hits,
            rerank_result,
            refusal_threshold=threshold,
        )

    @staticmethod
    def generate_from_reranked_hits(
        query: str,
        recall_hits: list[dict],
        rerank_hits: list[dict],
        *,
        refusal_threshold: float = DEFAULT_RERANK_REFUSAL_THRESHOLD,
    ) -> RAGPipelineResult:
        """离线评测专用：已有 rerank 分数时直接组装，无需实例化 Pipeline。"""
        return compose_from_reranked_hits(
            query,
            recall_hits,
            rerank_hits,
            refusal_threshold=refusal_threshold,
        )

    def _effective_config(self, overrides: RAGPipelineConfig | None = None) -> RAGPipelineConfig:
        if overrides is None:
            return self.config
        return overrides

    def _run_search_core(
        self,
        rag_query: RAGQuery,
        *,
        rewrite: QueryRewriteResult | None = None,
    ) -> RAGSearchResult:
        if rewrite is None:
            rewrite = self.query_rewrite(rag_query)
        retrieved = self.hybrid_retrieve(rewrite)
        reranked = self.rerank_step_for_rewrite(rewrite, retrieved.hits)
        return RAGSearchResult.from_steps(rewrite, retrieved, reranked)

    def _run_chat_core(
        self,
        rag_query: RAGQuery,
        cfg: RAGPipelineConfig,
        *,
        rewrite: QueryRewriteResult | None = None,
    ) -> RAGPipelineResult:
        if rewrite is None:
            rewrite = self.query_rewrite(rag_query)
        retrieved = self.hybrid_retrieve(rewrite)
        reranked = self.rerank_step_for_rewrite(rewrite, retrieved.hits)
        return compose_pipeline_result(
            rewrite.query,
            retrieved.hits,
            reranked,
            refusal_threshold=cfg.refusal_threshold,
            query_rewrite=rewrite,
            retrieve_result=retrieved,
        )

    def run_search(
        self,
        query: RAGQuery | str,
        *,
        config: RAGPipelineConfig | None = None,
        use_cache: bool = True,
    ) -> RAGSearchResult:
        """执行检索链路：query_rewrite → hybrid_retrieve → rerank（不含生成）。"""
        from cache.pipeline_bridge import run_search_with_cache

        cfg = self._effective_config(config)
        rag_query = self._normalize_query(query)
        with self._config_lock:
            saved = self.config
            self.config = cfg
            try:
                return run_search_with_cache(self, rag_query, cfg, use_cache=use_cache)
            finally:
                self.config = saved

    def run(
        self,
        query: RAGQuery | str,
        *,
        config: RAGPipelineConfig | None = None,
        use_cache: bool = True,
    ) -> RAGPipelineResult:
        """执行完整主链路：五步顺序编排。"""
        from cache.pipeline_bridge import run_chat_with_cache

        cfg = self._effective_config(config)
        rag_query = self._normalize_query(query)
        with self._config_lock:
            saved = self.config
            self.config = cfg
            try:
                return run_chat_with_cache(self, rag_query, cfg, use_cache=use_cache)
            finally:
                self.config = saved

    def retrieve_and_rerank(self, query: str) -> list[dict]:
        """兼容旧接口。"""
        rewrite = self.query_rewrite(query)
        retrieved = self.hybrid_retrieve(rewrite)
        return self.rerank_step_for_rewrite(rewrite, retrieved.hits).hits

    def answer(self, query: str) -> RAGPipelineResult:
        """兼容旧接口。"""
        return self.run(query)
