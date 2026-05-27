"""
BM25 稀疏检索索引（与 Milvus 向量库 chunk 集合对齐）。

索引文件：data/vector/bm25_index.pkl
构建：python src/build_bm25_index.py
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CHUNKS_JSONL = PROJECT_ROOT / "data" / "parsed" / "chunks.jsonl"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "vector" / "bm25_index.pkl"

# 检索字段与 embed 一致：embedding_text 优先
_JIEBA_USER_DICT_LOADED = False


def tokenize_zh(text: str) -> list[str]:
    global _JIEBA_USER_DICT_LOADED
    if not _JIEBA_USER_DICT_LOADED:
        jieba.initialize()
        _JIEBA_USER_DICT_LOADED = True
    text = text.strip()
    if not text:
        return []
    tokens = jieba.lcut_for_search(text)
    normalized: list[str] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if re.fullmatch(r"[\W_]+", token):
            continue
        normalized.append(token.lower() if token.isascii() else token)
    return normalized


def passage_text(record: dict) -> str:
    return str(record.get("embedding_text") or record.get("text") or "")


def load_retrievable_records(chunks_path: Path) -> list[dict]:
    records: list[dict] = []
    with open(chunks_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("is_retrievable", True):
                records.append(record)
    return records


class BM25ChunkIndex:
    def __init__(
        self,
        chunk_ids: list[str],
        corpus_tokens: list[list[str]],
        bm25: BM25Okapi,
        metadata_by_id: dict[str, dict],
    ) -> None:
        self.chunk_ids = chunk_ids
        self.corpus_tokens = corpus_tokens
        self.bm25 = bm25
        self.metadata_by_id = metadata_by_id

    @classmethod
    def build_from_chunks(cls, chunks_path: Path) -> BM25ChunkIndex:
        records = load_retrievable_records(chunks_path)
        chunk_ids: list[str] = []
        corpus_tokens: list[list[str]] = []
        metadata_by_id: dict[str, dict] = {}

        for record in records:
            chunk_id = record["chunk_id"]
            chunk_ids.append(chunk_id)
            corpus_tokens.append(tokenize_zh(passage_text(record)))
            metadata_by_id[chunk_id] = record

        bm25 = BM25Okapi(corpus_tokens)
        return cls(chunk_ids, corpus_tokens, bm25, metadata_by_id)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunk_ids": self.chunk_ids,
            "corpus_tokens": self.corpus_tokens,
            "metadata_by_id": self.metadata_by_id,
        }
        with open(path, "wb") as output_file:
            pickle.dump(payload, output_file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> BM25ChunkIndex:
        with open(path, "rb") as input_file:
            payload = pickle.load(input_file)
        chunk_ids = payload["chunk_ids"]
        corpus_tokens = payload["corpus_tokens"]
        metadata_by_id = payload["metadata_by_id"]
        bm25 = BM25Okapi(corpus_tokens)
        return cls(chunk_ids, corpus_tokens, bm25, metadata_by_id)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        query_tokens = tokenize_zh(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        hits: list[dict] = []
        for rank, index in enumerate(ranked_indices, start=1):
            chunk_id = self.chunk_ids[index]
            record = self.metadata_by_id[chunk_id]
            hits.append(
                {
                    "score": float(scores[index]),
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "doc_id": record.get("doc_id", ""),
                    "filename": record.get("filename", ""),
                    "display_name": record.get("display_name", ""),
                    "company_name": record.get("company_name", ""),
                    "report_title": record.get("report_title", ""),
                    "broker": record.get("broker", ""),
                    "industry_label": record.get("industry_label", ""),
                    "source_pdf_path": record.get("source_pdf_path", ""),
                    "section_title": record.get("section_title", ""),
                    "text": record.get("text", ""),
                    "page_start": record.get("page_start", 0),
                    "page_end": record.get("page_end", 0),
                    "contains_table": record.get("contains_table", False),
                    "stock_code": record.get("stock_code", ""),
                    "rating": record.get("rating", ""),
                }
            )
        return hits
