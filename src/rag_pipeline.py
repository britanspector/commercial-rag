"""
RAG 流水线：向量召回 → Rerank → 带引用生成 / 低分拒答。
"""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from embed_chunks import EMBED_DIM, OUTPUT_MILVUS_DB, load_embedder, resolve_device
from eval_retrieval import encode_query
from milvus_store import MilvusChunkStore
from rag_answer import RAGAnswer, generate_answer_with_citations
from rag_constants import (
    DEFAULT_RERANK_REFUSAL_THRESHOLD,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_RECALL_TOP_K,
    REFUSAL_MESSAGE,
)
from reranker import BGEReranker
from retrieval import DEFAULT_OUTPUT_FIELDS, _hits_from_vector

# 兼容旧 import
__all__ = [
    "RAGPipeline",
    "RAGAnswer",
    "generate_answer_with_citations",
    "REFUSAL_MESSAGE",
    "DEFAULT_RERANK_REFUSAL_THRESHOLD",
]


class RAGPipeline:
    def __init__(
        self,
        recall_top_k: int = DEFAULT_RECALL_TOP_K,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
        refusal_threshold: float = DEFAULT_RERANK_REFUSAL_THRESHOLD,
    ) -> None:
        self.recall_top_k = recall_top_k
        self.rerank_top_k = rerank_top_k
        self.refusal_threshold = refusal_threshold
        self._embedder = None
        self._store: MilvusChunkStore | None = None
        self._reranker: BGEReranker | None = None

    def _ensure_loaded(self) -> None:
        if self._embedder is None:
            device = resolve_device()
            self._embedder = load_embedder(device)
            self._store = MilvusChunkStore(OUTPUT_MILVUS_DB, vector_dim=EMBED_DIM)
            self._store.load()
            self._reranker = BGEReranker(device=device)

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
        self._store = None
        self._embedder = None
        self._reranker = None

    def retrieve_vector(self, query: str, top_k: int) -> list[dict]:
        self._ensure_loaded()
        assert self._embedder is not None and self._store is not None
        query_vector = encode_query(self._embedder, query)
        return _hits_from_vector(
            self._store, query_vector, top_k, DEFAULT_OUTPUT_FIELDS
        )

    def retrieve_and_rerank(self, query: str) -> list[dict]:
        self._ensure_loaded()
        assert self._reranker is not None
        candidates = self.retrieve_vector(query, self.recall_top_k)
        return self._reranker.rerank_hits(
            query, candidates, top_k=self.rerank_top_k, normalize=True
        )

    def answer(self, query: str) -> RAGAnswer:
        hits = self.retrieve_and_rerank(query)
        return generate_answer_with_citations(
            query, hits, refusal_threshold=self.refusal_threshold
        )
