"""
将 chunks.jsonl 向量化并写入 Milvus Lite。

输入：data/parsed/chunks.jsonl
输出：data/vector/milvus.db

模型：BAAI/bge-large-zh-v1.5（1024 维，与 chunk tokenizer 一致）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from milvus_store import MilvusChunkStore, chunk_record_to_milvus_row, reset_local_db

PROJECT_ROOT = CURRENT_DIR.parent

INPUT_CHUNKS_JSONL = PROJECT_ROOT / "data" / "parsed" / "chunks.jsonl"
OUTPUT_MILVUS_DB = PROJECT_ROOT / "data" / "vector" / "milvus.db"
OUTPUT_SUMMARY_CSV = PROJECT_ROOT / "data" / "parsed" / "embed_summary.csv"

EMBED_MODEL = "BAAI/bge-large-zh-v1.5"
EMBED_DIM = 1024
EMBED_BATCH_SIZE = 8

# 设备：auto / cuda / cpu（GPU 显存不足时可改为 cpu）
EMBED_DEVICE = "auto"
NORMALIZE_EMBEDDINGS = True

# bge 检索：查询侧加 instruction，文档侧不加
BGE_QUERY_PREFIX = "query: "


def resolve_device() -> str:
    if EMBED_DEVICE == "cpu":
        return "cpu"
    if EMBED_DEVICE == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_chunk_records() -> list[dict]:
    if not INPUT_CHUNKS_JSONL.exists():
        raise FileNotFoundError(
            f"未找到 chunks 文件，请先运行 src/chunk_mineru.py\n{INPUT_CHUNKS_JSONL}"
        )

    records: list[dict] = []
    with open(INPUT_CHUNKS_JSONL, "r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_embedder(device: str) -> SentenceTransformer:
    if device == "cuda":
        torch.cuda.empty_cache()
    try:
        return SentenceTransformer(EMBED_MODEL, device=device)
    except RuntimeError as error:
        if device == "cuda" and "out of memory" in str(error).lower():
            print("[警告] GPU 显存不足，Embedding 回退到 CPU")
            return SentenceTransformer(EMBED_MODEL, device="cpu")
        raise


def encode_passages(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    try:
        embeddings = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            print("[警告] GPU 显存不足，改用 CPU 重新 encode ...")
            cpu_model = SentenceTransformer(EMBED_MODEL, device="cpu")
            embeddings = cpu_model.encode(
                texts,
                batch_size=EMBED_BATCH_SIZE,
                normalize_embeddings=NORMALIZE_EMBEDDINGS,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        else:
            raise
    return embeddings.tolist()


def embed_and_store(recreate: bool = True) -> None:
    device = resolve_device()
    records = load_chunk_records()
    retrievable_records = [record for record in records if record.get("is_retrievable", True)]

    if not retrievable_records:
        raise ValueError("没有可检索的 chunk，请先运行 src/chunk_mineru.py。")

    print(f"总 chunk 数：{len(records)}，可检索：{len(retrievable_records)}")

    print("=" * 70)
    print("开始向量化 chunks 并写入 Milvus Lite")
    print(f"输入：{INPUT_CHUNKS_JSONL}")
    print(f"输出：{OUTPUT_MILVUS_DB}")
    print(f"Embedding 模型：{EMBED_MODEL}")
    print(f"设备：{device}")
    print(f"chunk 数量（可检索）：{len(retrievable_records)}")
    print("=" * 70)

    if recreate:
        print("清理并重建 Milvus Lite 本地库 ...")
        reset_local_db(OUTPUT_MILVUS_DB)

    model = load_embedder(device)

    texts = [
        record.get("embedding_text") or record["text"]
        for record in retrievable_records
    ]
    all_rows: list[dict] = []

    for start in tqdm(range(0, len(retrievable_records), EMBED_BATCH_SIZE), desc="Embedding"):
        batch_records = retrievable_records[start : start + EMBED_BATCH_SIZE]
        batch_texts = texts[start : start + EMBED_BATCH_SIZE]
        batch_vectors = encode_passages(model, batch_texts)
        all_rows.extend(
            chunk_record_to_milvus_row(record, vector)
            for record, vector in zip(batch_records, batch_vectors)
        )

    del model

    print(f"写入 Milvus（{len(all_rows)} 条）...")
    store = MilvusChunkStore(OUTPUT_MILVUS_DB, vector_dim=EMBED_DIM)
    try:
        if recreate or not store.has_collection():
            store.recreate_collection()
        store.insert_batch(all_rows)
        row_count = store.count()
    finally:
        store.close()

    summary = {
        "embed_model": EMBED_MODEL,
        "embed_dim": EMBED_DIM,
        "device": device,
        "chunk_count": len(retrievable_records),
        "total_chunk_count": len(records),
        "milvus_row_count": row_count,
        "milvus_db_path": str(OUTPUT_MILVUS_DB.relative_to(PROJECT_ROOT)),
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
    }
    pd.DataFrame([summary]).to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("向量化完成")
    print(f"写入向量数：{row_count}")
    print(f"Milvus Lite：{OUTPUT_MILVUS_DB}")
    print(f"统计文件：{OUTPUT_SUMMARY_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        embed_and_store(recreate=True)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        sys.exit(1)
