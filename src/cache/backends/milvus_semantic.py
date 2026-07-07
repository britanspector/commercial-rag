"""Milvus L2 语义缓存（向量近邻 + metadata 过滤）。"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from cache.backends.base import SemanticCacheBackend
from cache.config import CacheSettings, cache_settings
from cache.policy import is_entry_expired, normalize_query
from cache.types import (
    CacheEntry,
    CacheInvalidateFilter,
    CacheKey,
    CacheLayer,
    CacheMetadataFilters,
    CacheQueryContext,
    CacheScope,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag_semantic_cache"
METRIC_TYPE = "COSINE"
PRIMARY_KEY_MAX_LENGTH = 64


def _escape_filter(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _milvus_similarity(distance: float) -> float:
    """Milvus COSINE distance → cosine similarity。"""
    return 1.0 - float(distance)


def _extract_doc_ids(chunk_ids: list[str]) -> str:
    doc_ids: set[str] = set()
    for chunk_id in chunk_ids:
        cid = chunk_id.strip()
        if not cid:
            continue
        if "_c" in cid:
            doc_ids.add(cid.rsplit("_c", 1)[0])
        else:
            doc_ids.add(cid.split("_", 1)[0])
    return ",".join(sorted(doc_ids))


def make_cache_id(key: CacheKey, *, original_query: str, created_at_iso: str) -> str:
    raw = f"{key.semantic_bucket_key()}|{normalize_query(original_query)}|{created_at_iso}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:PRIMARY_KEY_MAX_LENGTH]


def _metadata_to_dict(meta: CacheMetadataFilters | None) -> dict[str, str]:
    if meta is None:
        return {}
    return {
        "stock_code": meta.stock_code,
        "query_type": meta.query_type,
        "company_name": meta.company_name,
        "report_year": meta.report_year,
        "doc_id": meta.doc_id,
        "doc_version": meta.doc_version,
    }


def _metadata_from_dict(data: dict[str, Any]) -> CacheMetadataFilters:
    return CacheMetadataFilters(
        stock_code=str(data.get("stock_code", "")),
        query_type=str(data.get("query_type", "factual")),
        company_name=str(data.get("company_name", "")),
        report_year=str(data.get("report_year", "")),
        doc_id=str(data.get("doc_id", "")),
        doc_version=str(data.get("doc_version", "")),
    )


def entry_to_row(entry: CacheEntry) -> dict[str, Any]:
    meta = entry.metadata_filters or CacheMetadataFilters(
        stock_code=entry.key.stock_code,
        query_type=entry.key.query_type,
    )
    cache_id = entry.cache_id or make_cache_id(
        entry.key,
        original_query=entry.original_query or entry.key.query_normalized,
        created_at_iso=entry.created_at_iso,
    )
    if not entry.query_embedding:
        raise ValueError("L2 entry requires query_embedding")

    return {
        "id": cache_id,
        "vector": entry.query_embedding,
        "bucket_key": entry.key.semantic_bucket_key(),
        "bucket_key_hash": entry.key.semantic_bucket_hash(),
        "scope": entry.key.scope.value,
        "stock_code": entry.key.stock_code,
        "query_type": entry.key.query_type,
        "config_fingerprint": entry.key.config_fingerprint,
        "index_fingerprint": entry.key.index_fingerprint,
        "generation_fingerprint": entry.key.generation_fingerprint,
        "original_query": entry.original_query or entry.key.query_normalized,
        "rewritten_query": entry.rewritten_query or entry.original_query or entry.key.query_normalized,
        "metadata_json": json.dumps(_metadata_to_dict(meta), ensure_ascii=False),
        "metadata_fingerprint": entry.key.metadata_filter_fingerprint or meta.fingerprint(),
        "payload_json": json.dumps(entry.payload, ensure_ascii=False),
        "chunk_ids_json": json.dumps(entry.chunk_ids, ensure_ascii=False),
        "doc_ids": _extract_doc_ids(entry.chunk_ids),
        "created_at_iso": entry.created_at_iso,
        "ttl_s": int(entry.ttl_s),
        "refused": bool(entry.refused),
        "top_rerank_score": float(entry.top_rerank_score),
    }


def row_to_entry(row: dict[str, Any], *, similarity: float | None = None) -> CacheEntry:
    meta_raw = row.get("metadata_json") or "{}"
    meta = _metadata_from_dict(json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw)
    chunk_ids_raw = row.get("chunk_ids_json") or "[]"
    chunk_ids = json.loads(chunk_ids_raw) if isinstance(chunk_ids_raw, str) else list(chunk_ids_raw or [])
    payload_raw = row.get("payload_json") or "{}"
    payload = json.loads(payload_raw) if isinstance(payload_raw, str) else dict(payload_raw or {})

    key = CacheKey(
        scope=CacheScope(str(row.get("scope", CacheScope.SEARCH.value))),
        query_normalized=normalize_query(str(row.get("original_query", ""))),
        stock_code=str(row.get("stock_code", "")),
        query_type=str(row.get("query_type", "factual")),
        config_fingerprint=str(row.get("config_fingerprint", "")),
        index_fingerprint=str(row.get("index_fingerprint", "")),
        generation_fingerprint=str(row.get("generation_fingerprint", "")),
        metadata_filter_fingerprint=str(row.get("metadata_fingerprint") or ""),
    )
    return CacheEntry(
        key=key,
        created_at_iso=str(row.get("created_at_iso", "")),
        ttl_s=int(row.get("ttl_s") or 0),
        query_embedding=None,
        payload=payload,
        chunk_ids=[str(x) for x in chunk_ids],
        refused=bool(row.get("refused", False)),
        top_rerank_score=float(row.get("top_rerank_score") or 0.0),
        exact_match=False,
        layer=CacheLayer.L2_SEMANTIC,
        original_query=str(row.get("original_query", "")),
        rewritten_query=str(row.get("rewritten_query", "")),
        metadata_filters=meta,
        semantic_similarity=similarity,
        cache_id=str(row.get("id", "")),
    )


class MilvusSemanticBackend(SemanticCacheBackend):
    """
    L2 Milvus 语义缓存：桶内向量检索 + TTL + metadata 条件过滤。

    Milvus 不可用时 lookup/store 降级为 no-op，不抛异常。
    """

    def __init__(
        self,
        *,
        db_path: str | Path,
        vector_dim: int = 1024,
        search_top_k: int = 5,
        similarity_threshold: float | None = None,
        settings: CacheSettings | None = None,
        client: Any | None = None,
    ) -> None:
        cfg = settings or cache_settings
        self._settings = cfg
        self._vector_dim = vector_dim
        self._search_top_k = max(1, search_top_k)
        self._similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else cfg.similarity_threshold
        )
        self._db_path = Path(db_path)
        self._client: Any | None = client
        self._available = False
        self._last_error: str | None = None

        if client is not None:
            self._available = True
            return

        try:
            from pymilvus import MilvusClient

            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._client = MilvusClient(uri=str(self._db_path))
            self._ensure_collection()
            self._available = True
            logger.info("Milvus L2 cache ready db=%s", self._db_path)
        except Exception as exc:
            self._last_error = str(exc)
            self._client = None
            self._available = False
            logger.warning("Milvus L2 cache unavailable, will degrade: %s", exc)

    @property
    def implemented(self) -> bool:
        return self._available and self._client is not None

    @property
    def available(self) -> bool:
        return self.implemented

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _ensure_collection(self) -> None:
        assert self._client is not None
        if self._client.has_collection(COLLECTION_NAME):
            return
        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=self._vector_dim,
            metric_type=METRIC_TYPE,
            id_type="string",
            max_length=PRIMARY_KEY_MAX_LENGTH,
        )

    def lookup(self, context: CacheQueryContext) -> CacheEntry | None:
        if not self.implemented or not context.query_embedding:
            return None
        assert self._client is not None

        bucket_hash = context.key.semantic_bucket_hash()
        filter_expr = f'bucket_key_hash == "{_escape_filter(bucket_hash)}"'
        try:
            results = self._client.search(
                collection_name=COLLECTION_NAME,
                data=[context.query_embedding],
                limit=self._search_top_k,
                filter=filter_expr,
                output_fields=[
                    "id",
                    "scope",
                    "stock_code",
                    "query_type",
                    "config_fingerprint",
                    "index_fingerprint",
                    "generation_fingerprint",
                    "original_query",
                    "rewritten_query",
                    "metadata_json",
                    "payload_json",
                    "chunk_ids_json",
                    "created_at_iso",
                    "ttl_s",
                    "refused",
                    "top_rerank_score",
                ],
                search_params={"metric_type": METRIC_TYPE},
            )
        except Exception as exc:
            self._handle_error("lookup", exc)
            return None

        if not results or not results[0]:
            return None

        best: CacheEntry | None = None
        best_sim = -1.0
        for hit in results[0]:
            entity = hit.get("entity", {})
            row = dict(entity)
            row["id"] = entity.get("id") or hit.get("id")
            similarity = _milvus_similarity(hit.get("distance", 1.0))
            if similarity < self._similarity_threshold:
                continue
            entry = row_to_entry(row, similarity=similarity)
            if is_entry_expired(entry):
                cache_id = entry.cache_id
                if cache_id:
                    self._delete_by_ids([cache_id])
                continue
            if similarity > best_sim:
                best = entry
                best_sim = similarity

        return best

    def put(self, entry: CacheEntry) -> None:
        if not self.implemented or not entry.query_embedding:
            return None
        assert self._client is not None
        try:
            row = entry_to_row(entry)
            self._client.insert(collection_name=COLLECTION_NAME, data=[row])
        except Exception as exc:
            self._handle_error("put", exc)
        return None

    def invalidate(self, filter_: CacheInvalidateFilter) -> int:
        if not self.implemented:
            return 0
        assert self._client is not None
        try:
            if filter_.all_entries:
                if not self._client.has_collection(COLLECTION_NAME):
                    return 0
                stats = self._client.get_collection_stats(COLLECTION_NAME)
                count = int(stats.get("row_count", 0))
                self._client.drop_collection(COLLECTION_NAME)
                self._ensure_collection()
                return count

            expr_parts: list[str] = []
            if filter_.scope is not None:
                expr_parts.append(f'scope == "{_escape_filter(filter_.scope.value)}"')
            if filter_.index_fingerprint is not None:
                expr_parts.append(
                    f'index_fingerprint == "{_escape_filter(filter_.index_fingerprint)}"'
                )
            if filter_.doc_id:
                doc_id = _escape_filter(filter_.doc_id.strip())
                expr_parts.append(f'doc_ids like "%{doc_id}%"')

            if not expr_parts:
                return 0

            filter_expr = " and ".join(expr_parts)
            result = self._client.delete(collection_name=COLLECTION_NAME, filter=filter_expr)
            delete_count = result.get("delete_count") if isinstance(result, dict) else 0
            return int(delete_count or 0)
        except Exception as exc:
            self._handle_error("invalidate", exc)
            return 0

    def count(self) -> int:
        if not self.implemented:
            return 0
        assert self._client is not None
        try:
            if not self._client.has_collection(COLLECTION_NAME):
                return 0
            stats = self._client.get_collection_stats(COLLECTION_NAME)
            return int(stats.get("row_count", 0))
        except Exception:
            return 0

    def ping(self) -> bool:
        if not self._client:
            return False
        try:
            if not self._client.has_collection(COLLECTION_NAME):
                self._ensure_collection()
            return True
        except Exception as exc:
            self._last_error = str(exc)
            self._available = False
            return False

    def delete_cache_id(self, cache_id: str) -> bool:
        if not self.implemented or not cache_id.strip():
            return False
        try:
            self._delete_by_ids([cache_id.strip()])
            return True
        except Exception as exc:
            self._handle_error("delete_cache_id", exc)
            return False

    def _delete_by_ids(self, ids: list[str]) -> None:
        if not ids or not self._client:
            return
        quoted = ", ".join(f'"{_escape_filter(i)}"' for i in ids)
        self._client.delete(collection_name=COLLECTION_NAME, filter=f"id in [{quoted}]")

    def _handle_error(self, op: str, exc: Exception) -> None:
        self._last_error = str(exc)
        logger.warning("Milvus L2 %s failed (degraded to miss/no-op): %s", op, exc)
