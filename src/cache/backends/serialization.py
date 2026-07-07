"""CacheEntry JSON 序列化（Redis L1 存储）。"""

from __future__ import annotations

import json
from typing import Any

from cache.types import CacheEntry, CacheKey, CacheLayer, CacheMetadataFilters, CacheScope


def _metadata_filters_to_dict(meta: CacheMetadataFilters | None) -> dict[str, str]:
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


def _metadata_filters_from_dict(data: dict) -> CacheMetadataFilters | None:
    if not data:
        return None
    return CacheMetadataFilters(
        stock_code=str(data.get("stock_code", "")),
        query_type=str(data.get("query_type", "factual")),
        company_name=str(data.get("company_name", "")),
        report_year=str(data.get("report_year", "")),
        doc_id=str(data.get("doc_id", "")),
        doc_version=str(data.get("doc_version", "")),
    )


def _key_to_dict(key: CacheKey) -> dict[str, str]:
    return {
        "scope": key.scope.value,
        "query_normalized": key.query_normalized,
        "stock_code": key.stock_code,
        "query_type": key.query_type,
        "config_fingerprint": key.config_fingerprint,
        "index_fingerprint": key.index_fingerprint,
        "generation_fingerprint": key.generation_fingerprint,
        "metadata_filter_fingerprint": key.metadata_filter_fingerprint,
    }


def _key_from_dict(data: dict[str, Any]) -> CacheKey:
    return CacheKey(
        scope=CacheScope(str(data["scope"])),
        query_normalized=str(data.get("query_normalized", "")),
        stock_code=str(data.get("stock_code", "")),
        query_type=str(data.get("query_type", "factual")),
        config_fingerprint=str(data.get("config_fingerprint", "")),
        index_fingerprint=str(data.get("index_fingerprint", "")),
        generation_fingerprint=str(data.get("generation_fingerprint", "")),
        metadata_filter_fingerprint=str(data.get("metadata_filter_fingerprint", "")),
    )


def entry_to_json(entry: CacheEntry) -> str:
    payload = {
        "v": 1,
        "key": _key_to_dict(entry.key),
        "created_at_iso": entry.created_at_iso,
        "ttl_s": entry.ttl_s,
        "payload": entry.payload,
        "chunk_ids": entry.chunk_ids,
        "refused": entry.refused,
        "top_rerank_score": entry.top_rerank_score,
        "exact_match": entry.exact_match,
        "source_request_id": entry.source_request_id,
        "layer": entry.layer.value,
        "storage_key": entry.key.storage_key(),
        "metadata_filters": _metadata_filters_to_dict(entry.metadata_filters),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def entry_from_json(raw: str | bytes) -> CacheEntry:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    data = json.loads(raw)
    layer_raw = data.get("layer", CacheLayer.L1_EXACT.value)
    return CacheEntry(
        key=_key_from_dict(data["key"]),
        created_at_iso=str(data["created_at_iso"]),
        ttl_s=int(data["ttl_s"]),
        query_embedding=None,
        payload=dict(data.get("payload") or {}),
        chunk_ids=[str(x) for x in (data.get("chunk_ids") or [])],
        refused=bool(data.get("refused", False)),
        top_rerank_score=float(data.get("top_rerank_score") or 0.0),
        exact_match=bool(data.get("exact_match", True)),
        source_request_id=data.get("source_request_id"),
        layer=CacheLayer(str(layer_raw)),
        metadata_filters=_metadata_filters_from_dict(data.get("metadata_filters") or {}),
    )
