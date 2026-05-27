"""
三路召回：纯向量 / 纯 BM25 / 混合（加权融合）。

路线 A — vector：Milvus COSINE
路线 B — bm25：BM25Okapi + jieba
路线 C — hybrid：min-max 归一化后 score = w_vec * V + w_bm25 * B
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from bm25_store import BM25ChunkIndex, DEFAULT_INDEX_PATH
from milvus_store import MilvusChunkStore

DEFAULT_OUTPUT_FIELDS = [
    "chunk_id",
    "doc_id",
    "filename",
    "display_name",
    "company_name",
    "report_title",
    "broker",
    "industry_label",
    "source_pdf_path",
    "section_title",
    "text",
    "page_start",
    "page_end",
    "contains_table",
    "stock_code",
    "rating",
]

# 混合检索默认权重：向量 0.5 + BM25 0.5
DEFAULT_HYBRID_VECTOR_WEIGHT = 0.5
# 融合候选池（两路各取 pool 再合并）
DEFAULT_HYBRID_POOL_SIZE = 100


class RecallRoute(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return {key: 1.0 for key in scores}
    span = maximum - minimum
    return {key: (value - minimum) / span for key, value in scores.items()}


def _distance_to_similarity(distance: float) -> float:
    """Milvus COSINE 返回的是距离（越小越相似），转为越大越好的相似度。"""
    return 1.0 - distance


def _hits_from_vector(
    store: MilvusChunkStore,
    query_vector: list[float],
    top_k: int,
    output_fields: list[str],
) -> list[dict]:
    hits = store.search(query_vector, top_k=top_k, output_fields=output_fields)
    for hit in hits:
        distance = float(hit.get("score") or 0.0)
        hit["score_distance"] = distance
        hit["score_vector"] = _distance_to_similarity(distance)
        hit["score"] = hit["score_vector"]
    return hits


def _hits_from_bm25(index: BM25ChunkIndex, query: str, top_k: int) -> list[dict]:
    hits = index.search(query, top_k=top_k)
    for hit in hits:
        hit["score_bm25"] = float(hit.get("score") or 0.0)
    return hits


def _merge_hybrid(
    vector_hits: list[dict],
    bm25_hits: list[dict],
    top_k: int,
    vector_weight: float,
) -> list[dict]:
    vector_scores = {
        hit["chunk_id"]: float(hit.get("score_vector") or hit.get("score") or 0.0)
        for hit in vector_hits
    }
    bm25_scores = {
        hit["chunk_id"]: float(hit.get("score_bm25") or hit.get("score") or 0.0)
        for hit in bm25_hits
    }

    norm_vector = _min_max_normalize(vector_scores)
    norm_bm25 = _min_max_normalize(bm25_scores)
    bm25_weight = 1.0 - vector_weight

    all_chunk_ids = set(norm_vector) | set(norm_bm25)
    fused: list[tuple[str, float]] = []
    for chunk_id in all_chunk_ids:
        combined = vector_weight * norm_vector.get(chunk_id, 0.0) + bm25_weight * norm_bm25.get(
            chunk_id, 0.0
        )
        fused.append((chunk_id, combined))

    fused.sort(key=lambda item: item[1], reverse=True)
    top_ids = [chunk_id for chunk_id, _ in fused[:top_k]]

    hit_by_id: dict[str, dict] = {}
    for hit in vector_hits + bm25_hits:
        chunk_id = hit["chunk_id"]
        if chunk_id not in hit_by_id:
            hit_by_id[chunk_id] = dict(hit)

    merged_hits: list[dict] = []
    for chunk_id in top_ids:
        hit = dict(hit_by_id[chunk_id])
        hit["score"] = next(score for cid, score in fused if cid == chunk_id)
        hit["score_vector"] = vector_scores.get(chunk_id, 0.0)
        hit["score_bm25"] = bm25_scores.get(chunk_id, 0.0)
        merged_hits.append(hit)
    return merged_hits


class HybridRetriever:
    def __init__(
        self,
        milvus_store: MilvusChunkStore,
        bm25_index: BM25ChunkIndex,
        hybrid_vector_weight: float = DEFAULT_HYBRID_VECTOR_WEIGHT,
        hybrid_pool_size: int = DEFAULT_HYBRID_POOL_SIZE,
        output_fields: list[str] | None = None,
    ) -> None:
        self.milvus_store = milvus_store
        self.bm25_index = bm25_index
        self.hybrid_vector_weight = hybrid_vector_weight
        self.hybrid_pool_size = hybrid_pool_size
        self.output_fields = output_fields or DEFAULT_OUTPUT_FIELDS

    @classmethod
    def from_paths(
        cls,
        milvus_db: Path,
        vector_dim: int,
        bm25_index_path: Path = DEFAULT_INDEX_PATH,
        **kwargs,
    ) -> HybridRetriever:
        store = MilvusChunkStore(milvus_db, vector_dim=vector_dim)
        if not bm25_index_path.exists():
            raise FileNotFoundError(
                f"未找到 BM25 索引，请先运行 src/build_bm25_index.py\n{bm25_index_path}"
            )
        index = BM25ChunkIndex.load(bm25_index_path)
        return cls(store, index, **kwargs)

    def close(self) -> None:
        self.milvus_store.close()

    def retrieve(
        self,
        route: RecallRoute,
        query: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[dict]:
        if route == RecallRoute.VECTOR:
            return _hits_from_vector(
                self.milvus_store, query_vector, top_k, self.output_fields
            )

        if route == RecallRoute.BM25:
            return _hits_from_bm25(self.bm25_index, query, top_k)

        pool = max(top_k, self.hybrid_pool_size)
        vector_hits = _hits_from_vector(
            self.milvus_store, query_vector, pool, self.output_fields
        )
        bm25_hits = _hits_from_bm25(self.bm25_index, query, pool)
        return _merge_hybrid(
            vector_hits,
            bm25_hits,
            top_k,
            self.hybrid_vector_weight,
        )
