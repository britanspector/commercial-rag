"""
Milvus Lite 检索 smoke test。

用法：
    python src/check_milvus.py
    python src/check_milvus.py "华峰测控2025年营收是多少"
"""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from embed_chunks import (
    BGE_QUERY_PREFIX,
    EMBED_DIM,
    EMBED_MODEL,
    NORMALIZE_EMBEDDINGS,
    OUTPUT_MILVUS_DB,
    load_embedder,
    resolve_device,
)
from milvus_store import MilvusChunkStore

DEFAULT_QUERY = "华峰测控2025年营收和净利润是多少？"
TOP_K = 5
PREVIEW_CHARS = 400


def encode_query(model, query: str) -> list[float]:
    query_text = query if query.startswith(BGE_QUERY_PREFIX) else f"{BGE_QUERY_PREFIX}{query}"
    vector = model.encode(
        [query_text],
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        convert_to_numpy=True,
    )[0]
    return vector.tolist()


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY

    if not OUTPUT_MILVUS_DB.exists():
        raise FileNotFoundError(
            f"未找到 Milvus 数据库，请先运行 src/embed_chunks.py\n{OUTPUT_MILVUS_DB}"
        )

    device = resolve_device()
    print(f"Embedding 模型：{EMBED_MODEL}")
    print(f"设备：{device}")

    # 先完成 query 向量化，再打开 Milvus（Windows 上更稳定）
    model = load_embedder(device)
    query_vector = encode_query(model, query)
    del model

    store = MilvusChunkStore(OUTPUT_MILVUS_DB, vector_dim=EMBED_DIM)
    if not store.has_collection():
        store.close()
        raise FileNotFoundError("Milvus collection 不存在，请先运行 src/embed_chunks.py")

    row_count = store.count()
    store.load()
    print(f"Milvus 向量数：{row_count}")
    print(f"查询：{query}")
    print("=" * 90)

    hits = store.search(query_vector, top_k=TOP_K)
    store.close()

    if not hits:
        print("未检索到结果。")
        return

    for index, hit in enumerate(hits, start=1):
        print(f"\n[{index}] score={hit['score']:.4f}")
        print(f"chunk_id：{hit['chunk_id']}")
        print(f"文档：{hit.get('display_name') or hit['filename']}")
        if hit.get("company_name"):
            print(f"公司：{hit['company_name']} ({hit.get('stock_code', '')})")
        if hit.get("broker"):
            print(f"券商：{hit['broker']}")
        print(f"章节：{hit.get('section_title', '')}")
        print(f"页码：{hit.get('page_start')} - {hit.get('page_end')}")
        print(f"含表格：{hit.get('contains_table')}")
        print("\n【正文预览】")
        print((hit.get("text") or "")[:PREVIEW_CHARS])


if __name__ == "__main__":
    main()
