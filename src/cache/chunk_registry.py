"""Corpus chunk 存在性校验（防止返回已删除文档的引用）。"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from bm25_store import DEFAULT_CHUNKS_JSONL

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_chunk_ids: set[str] | None = None
_chunks_mtime: float = 0.0
_test_override: set[str] | None = None


def _load_chunk_ids(force: bool = False) -> set[str]:
    global _chunk_ids, _chunks_mtime, _test_override
    if _test_override is not None and not force:
        return _test_override

    path = DEFAULT_CHUNKS_JSONL
    if not path.is_file():
        return set()

    mtime = path.stat().st_mtime
    with _lock:
        if not force and _chunk_ids is not None and mtime == _chunks_mtime:
            return _chunk_ids

        ids: set[str] = set()
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(str(json.loads(line)["chunk_id"]))
                except (json.JSONDecodeError, KeyError):
                    continue
        _chunk_ids = ids
        _chunks_mtime = mtime
        return ids


def reset_chunk_registry() -> None:
    """测试或 upload 后强制刷新。"""
    global _chunk_ids, _chunks_mtime, _test_override
    with _lock:
        _chunk_ids = None
        _chunks_mtime = 0.0
        _test_override = None


def register_test_chunk_ids(chunk_ids: set[str]) -> None:
    """自测用：注入已知 chunk 集合，跳过读取 chunks.jsonl。"""
    global _test_override
    with _lock:
        _test_override = set(chunk_ids)


def verify_chunks_exist(chunk_ids: list[str]) -> tuple[bool, str, list[str]]:
    """
    校验缓存 entry 引用的 chunk 是否仍在 Corpus 中。

    Returns:
        (ok, reason, missing_ids)
    """
    if not chunk_ids:
        return True, "", []

    known = _load_chunk_ids()
    if not known:
        return True, "", []

    missing = [cid for cid in chunk_ids if cid not in known]
    if missing:
        return False, "chunk_missing", missing
    return True, "", []
