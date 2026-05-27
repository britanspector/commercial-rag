"""
RAG 问答 CLI：向量召回 → Rerank → 带引用回答 / 低分拒答。

用法：
    python src/rag_chat.py "京仪装备2026E毛利率预测是多少？"
    python src/rag_chat.py   # 交互模式
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from rag_constants import DEFAULT_RERANK_REFUSAL_THRESHOLD
from rag_pipeline import RAGPipeline


def print_answer(result) -> None:
    print("\n" + "=" * 70)
    if result.refused:
        print(f"[拒答] {result.refusal_reason} | top_rerank={result.top_rerank_score:.3f}")
    else:
        print(f"[回答] top_rerank={result.top_rerank_score:.3f}")
    print("-" * 70)
    print(result.answer)
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="研报 RAG 问答（引用溯源 + 拒答）")
    parser.add_argument("query", nargs="?", help="问题（省略则进入交互）")
    parser.add_argument("--recall-top-k", type=int, default=20)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument(
        "--refusal-threshold",
        type=float,
        default=DEFAULT_RERANK_REFUSAL_THRESHOLD,
    )
    args = parser.parse_args()

    pipeline = RAGPipeline(
        recall_top_k=args.recall_top_k,
        rerank_top_k=args.rerank_top_k,
        refusal_threshold=args.refusal_threshold,
    )

    try:
        if args.query:
            print_answer(pipeline.answer(args.query))
            return

        print("研报 RAG 问答（输入空行或 quit 退出）")
        while True:
            try:
                query = input("\n问题> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query or query.lower() in {"quit", "exit", "q"}:
                break
            print_answer(pipeline.answer(query))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
