"""索引 / Corpus 版本指纹（用于 cache 失效）。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MILVUS_MANIFEST = (
    PROJECT_ROOT / "data" / "vector" / "milvus.db" / "collections" / "rag_chunks" / "manifest.json"
)
DEFAULT_CHUNKS_JSONL = PROJECT_ROOT / "data" / "parsed" / "chunks.jsonl"
DEFAULT_BM25_INDEX = PROJECT_ROOT / "data" / "vector" / "bm25_index.pkl"
DEFAULT_DOC_MANIFEST = PROJECT_ROOT / "data" / "parsed" / "documents.jsonl"


def _count_jsonl_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def compute_index_fingerprint(
    *,
    milvus_manifest_path: Path | None = None,
    chunks_jsonl_path: Path | None = None,
    bm25_index_path: Path | None = None,
    doc_manifest_path: Path | None = None,
) -> str:
    """
    Corpus 版本指纹：Milvus 行数 + chunks/doc manifest + BM25 mtime。

    任一索引/文档集变更后 fingerprint 变化，旧 entry 不再 serve。
    """
    manifest_path = milvus_manifest_path or DEFAULT_MILVUS_MANIFEST
    chunks_path = chunks_jsonl_path or DEFAULT_CHUNKS_JSONL
    bm25_path = bm25_index_path or DEFAULT_BM25_INDEX
    docs_path = doc_manifest_path or DEFAULT_DOC_MANIFEST

    parts: list[str] = []

    if manifest_path.is_file():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            row_count = data.get("row_count", data.get("num_entities", ""))
            parts.append(f"milvus_rows={row_count}")
            parts.append(f"milvus_mtime={int(manifest_path.stat().st_mtime)}")
        except (OSError, json.JSONDecodeError):
            parts.append(f"milvus_mtime={int(manifest_path.stat().st_mtime)}")
    else:
        parts.append("milvus=missing")

    chunk_lines = _count_jsonl_lines(chunks_path)
    if chunks_path.is_file():
        parts.append(f"chunks_lines={chunk_lines}")
        parts.append(f"chunks_mtime={int(chunks_path.stat().st_mtime)}")
    else:
        parts.append("chunks=missing")

    doc_lines = _count_jsonl_lines(docs_path)
    if docs_path.is_file():
        parts.append(f"docs_lines={doc_lines}")
        parts.append(f"docs_mtime={int(docs_path.stat().st_mtime)}")
    else:
        parts.append("docs=missing")

    if bm25_path.is_file():
        parts.append(f"bm25_mtime={int(bm25_path.stat().st_mtime)}")
        try:
            parts.append(f"bm25_size={bm25_path.stat().st_size}")
        except OSError:
            pass
    else:
        parts.append("bm25=missing")

    raw = "|".join(parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{raw}|fp={digest}"
