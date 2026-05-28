"""
三路召回：纯向量 / 纯 BM25 / 混合（加权融合）。

路线 A — vector：Milvus COSINE
路线 B — bm25：BM25Okapi + jieba
路线 C — hybrid：min-max 归一化后 score = w_vec * V + w_bm25 * B

P1 增强：BM25 查询扩展、stock_code 加分、对比型多查询 RRF、更大候选池。
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from bm25_store import BM25ChunkIndex, DEFAULT_INDEX_PATH
from milvus_store import MilvusChunkStore
from query_enhance import (
    enhance_bm25_query,
    extract_compare_entities,
    hybrid_vector_weight,
)

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

# 混合检索：事实型略偏 BM25；候选池加大
DEFAULT_HYBRID_VECTOR_WEIGHT = 0.35
DEFAULT_HYBRID_POOL_SIZE = 200
STOCK_CODE_SCORE_BOOST = 0.12
COMPARABLE_TABLE_SCORE_PENALTY = 0.10
RRF_K = 60


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


def _union_bm25_priority(
    merged: list[dict],
    bm25_hits: list[dict],
    top_k: int,
) -> list[dict]:
    """保留 BM25 头部结果，避免向量分稀释稀疏关键词命中。"""
    seen = {hit["chunk_id"] for hit in merged}
    combined = list(merged)
    for hit in bm25_hits:
        chunk_id = hit["chunk_id"]
        if chunk_id in seen:
            continue
        combined.append(hit)
        seen.add(chunk_id)
        if len(combined) >= top_k:
            break
    return combined[:top_k]


def _enrich_hits_from_bm25_metadata(
    hits: list[dict],
    bm25_index: BM25ChunkIndex,
) -> None:
    for hit in hits:
        meta = bm25_index.metadata_by_id.get(hit.get("chunk_id", ""), {})
        if not meta:
            continue
        hit.setdefault("content_type", meta.get("content_type", ""))
        if not hit.get("embedding_text"):
            hit["embedding_text"] = meta.get("embedding_text") or meta.get("text", "")


def _apply_content_type_adjustments(hits: list[dict]) -> list[dict]:
    """P2：可比公司估值表降权，避免压过本公司盈利预测表。"""
    for hit in hits:
        content_type = str(hit.get("content_type", ""))
        if content_type == "comparable_table":
            hit["score"] = max(0.0, float(hit.get("score") or 0.0) - COMPARABLE_TABLE_SCORE_PENALTY)
    hits.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return hits


def _apply_stock_boost(hits: list[dict], stock_codes: list[str]) -> list[dict]:
    codes = {code.strip() for code in stock_codes if code and code.strip()}
    if not codes:
        return hits
    for hit in hits:
        if str(hit.get("stock_code", "")).strip() in codes:
            hit["score"] = float(hit.get("score") or 0.0) + STOCK_CODE_SCORE_BOOST
    hits.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return hits


def rrf_fuse(hit_lists: list[list[dict]], top_k: int, rrf_k: int = RRF_K) -> list[dict]:
    """多路召回 Reciprocal Rank Fusion。"""
    if not hit_lists:
        return []
    if len(hit_lists) == 1:
        return hit_lists[0][:top_k]

    scores: dict[str, float] = {}
    hit_by_id: dict[str, dict] = {}
    for hits in hit_lists:
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            hit_by_id[chunk_id] = hit

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    fused: list[dict] = []
    for chunk_id, score in ranked:
        hit = dict(hit_by_id[chunk_id])
        hit["score"] = score
        fused.append(hit)
    return fused


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

    def _retrieve_hybrid_once(
        self,
        query: str,
        query_vector: list[float],
        top_k: int,
        stock_code: str,
        query_type: str,
    ) -> list[dict]:
        pool = max(top_k, self.hybrid_pool_size)
        vector_weight = hybrid_vector_weight(
            query_type, query, default=self.hybrid_vector_weight
        )
        bm25_query = enhance_bm25_query(query, stock_code)

        vector_hits = _hits_from_vector(
            self.milvus_store, query_vector, pool, self.output_fields
        )
        bm25_hits = _hits_from_bm25(self.bm25_index, bm25_query, pool)
        merged = _merge_hybrid(vector_hits, bm25_hits, top_k, vector_weight)
        merged = _union_bm25_priority(merged, bm25_hits, top_k)
        _enrich_hits_from_bm25_metadata(merged, self.bm25_index)
        merged = _apply_content_type_adjustments(merged)
        boost_codes = [stock_code] if stock_code else []
        return _apply_stock_boost(merged, boost_codes)

    def retrieve(
        self,
        route: RecallRoute,
        query: str,
        query_vector: list[float],
        top_k: int,
        *,
        stock_code: str = "",
        query_type: str = "factual",
    ) -> list[dict]:
        if route == RecallRoute.VECTOR:
            hits = _hits_from_vector(
                self.milvus_store, query_vector, top_k, self.output_fields
            )
            _enrich_hits_from_bm25_metadata(hits, self.bm25_index)
            hits = _apply_content_type_adjustments(hits)
            return _apply_stock_boost(hits, [stock_code] if stock_code else [])

        if route == RecallRoute.BM25:
            bm25_query = enhance_bm25_query(query, stock_code)
            hits = _hits_from_bm25(self.bm25_index, bm25_query, top_k)
            hits = _apply_content_type_adjustments(hits)
            return _apply_stock_boost(hits, [stock_code] if stock_code else [])

        if query_type == "comparative":
            entities = extract_compare_entities(query)
            if len(entities) >= 2:
                pool = max(top_k, self.hybrid_pool_size)
                hit_lists: list[list[dict]] = []
                for entity in entities[:3]:
                    sub_query = f"{entity} {query}"
                    sub_vector = query_vector
                    hit_lists.append(
                        self._retrieve_hybrid_once(
                            sub_query, sub_vector, pool, stock_code="", query_type="factual"
                        )
                    )
                fused = rrf_fuse(hit_lists, top_k)
                _enrich_hits_from_bm25_metadata(fused, self.bm25_index)
                fused = _apply_content_type_adjustments(fused)
                return _apply_stock_boost(fused, [stock_code] if stock_code else [])

        return self._retrieve_hybrid_once(
            query, query_vector, top_k, stock_code, query_type
        )
