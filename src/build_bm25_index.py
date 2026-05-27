"""
从 chunks.jsonl 构建 BM25 索引。

用法：
    python src/build_bm25_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from bm25_store import BM25ChunkIndex, DEFAULT_CHUNKS_JSONL, DEFAULT_INDEX_PATH


def main() -> None:
    if not DEFAULT_CHUNKS_JSONL.exists():
        raise FileNotFoundError(
            f"未找到 chunks，请先运行 src/chunk_mineru.py\n{DEFAULT_CHUNKS_JSONL}"
        )

    print(f"构建 BM25 索引：{DEFAULT_CHUNKS_JSONL}")
    index = BM25ChunkIndex.build_from_chunks(DEFAULT_CHUNKS_JSONL)
    index.save(DEFAULT_INDEX_PATH)
    print(f"完成：{len(index.chunk_ids)} 条 → {DEFAULT_INDEX_PATH}")


if __name__ == "__main__":
    main()
