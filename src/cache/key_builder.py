"""从 RAG 请求上下文构造 CacheKey / CacheQueryContext。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cache.index_fingerprint import compute_index_fingerprint
from cache.policy import build_config_fingerprint, build_generation_fingerprint, normalize_query
from cache.types import CacheKey, CacheMetadataFilters, CacheQueryContext, CacheScope

if TYPE_CHECKING:
    from rag_pipeline import RAGPipelineConfig
    from rag_types import RAGQuery


DEFAULT_PROMPT_VERSION = "v1"


def build_cache_key(
    *,
    scope: CacheScope,
    query: str,
    stock_code: str = "",
    query_type: str = "factual",
    config_fingerprint: str,
    index_fingerprint: str | None = None,
    generation_fingerprint: str = "",
    metadata_filter_fingerprint: str = "",
) -> CacheKey:
    return CacheKey(
        scope=scope,
        query_normalized=normalize_query(query),
        stock_code=(stock_code or "").strip(),
        query_type=query_type or "factual",
        config_fingerprint=config_fingerprint,
        index_fingerprint=index_fingerprint or compute_index_fingerprint(),
        generation_fingerprint=generation_fingerprint,
        metadata_filter_fingerprint=metadata_filter_fingerprint,
    )


def build_config_fingerprint_from_pipeline(config: RAGPipelineConfig) -> str:
    return build_config_fingerprint(
        recall_route=config.recall_route.value
        if hasattr(config.recall_route, "value")
        else str(config.recall_route),
        recall_top_k=config.recall_top_k,
        rerank_top_k=config.rerank_top_k,
        refusal_threshold=config.refusal_threshold,
        hybrid_vector_weight=config.hybrid_vector_weight,
        hybrid_pool_size=config.hybrid_pool_size,
    )


def build_generation_fingerprint_from_env(
    *,
    llm_model: str | None = None,
    prompt_version: str | None = None,
) -> str:
    from generation_config import resolve_generation_config
    from pipeline.llm_prompts import PROMPT_VERSION

    cfg = resolve_generation_config()
    if llm_model is None:
        llm_model = cfg.llm_model
    return build_generation_fingerprint(
        llm_model=llm_model,
        prompt_version=prompt_version or PROMPT_VERSION,
        num_ctx=cfg.num_ctx,
        num_predict=cfg.num_predict,
    )


def build_query_context(
    *,
    scope: CacheScope,
    rag_query: RAGQuery,
    config_fingerprint: str,
    index_fingerprint: str | None = None,
    generation_fingerprint: str = "",
    query_embedding: list[float] | None = None,
    original_query: str = "",
    rewritten_query: str = "",
    metadata_filters: CacheMetadataFilters | None = None,
) -> CacheQueryContext:
    orig = original_query or rag_query.query
    meta = metadata_filters or CacheMetadataFilters(
        stock_code=rag_query.stock_code,
        query_type=rag_query.query_type,
    )
    key = build_cache_key(
        scope=scope,
        query=rag_query.query,
        stock_code=rag_query.stock_code,
        query_type=rag_query.query_type,
        config_fingerprint=config_fingerprint,
        index_fingerprint=index_fingerprint,
        generation_fingerprint=generation_fingerprint if scope == CacheScope.CHAT else "",
        metadata_filter_fingerprint=meta.fingerprint(),
    )
    return CacheQueryContext(
        key=key,
        query_embedding=query_embedding,
        original_query=orig,
        rewritten_query=rewritten_query or orig,
        metadata_filters=meta,
    )


def extract_chunk_ids(payload: dict) -> list[str]:
    """从 Pipeline 结果 dict 提取 chunk_id 列表（供失效与校验）。"""
    ids: list[str] = []
    seen: set[str] = set()

    def _add(value: object) -> None:
        if not value:
            return
        chunk_id = str(value).strip()
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            ids.append(chunk_id)

    for citation in payload.get("citations") or []:
        if isinstance(citation, dict):
            _add(citation.get("chunk_id"))

    for hit in payload.get("rerank_hits") or []:
        if isinstance(hit, dict):
            _add(hit.get("chunk_id"))

    rerank = payload.get("rerank") or {}
    if isinstance(rerank, dict):
        for hit in rerank.get("hits") or []:
            if isinstance(hit, dict):
                _add(hit.get("chunk_id"))

    recall = payload.get("recall") or {}
    if isinstance(recall, dict):
        for hit in recall.get("hits") or []:
            if isinstance(hit, dict):
                _add(hit.get("chunk_id"))

    return ids
