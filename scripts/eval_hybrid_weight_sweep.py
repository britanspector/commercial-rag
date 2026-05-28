"""
混合检索向量权重分组评测（BM25 权重 = 1 - vector_weight）。

用法：
    python scripts/eval_hybrid_weight_sweep.py
    python scripts/eval_hybrid_weight_sweep.py --weights 0.5 0.4 0.3 0.6 0.7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embed_chunks import EMBED_DIM, OUTPUT_MILVUS_DB, load_embedder, resolve_device
from eval_retrieval import (
    DEFAULT_QUESTIONS,
    aggregate_metrics,
    encode_query,
    evaluate_hits,
    load_questions,
)
from retrieval import HybridRetriever, RecallRoute

OUTPUT_DIR = ROOT / "data" / "eval"
DEFAULT_WEIGHTS = [0.5, 0.4, 0.35, 0.3, 0.6, 0.7]


def run_sweep(
    questions_path: Path,
    weights: list[float],
    top_k: int = 10,
) -> pd.DataFrame:
    questions = load_questions(questions_path)
    device = resolve_device()
    embedder = load_embedder(device)
    retriever = HybridRetriever.from_paths(OUTPUT_MILVUS_DB, vector_dim=EMBED_DIM)
    retriever.milvus_store.load()

    rows: list[dict] = []
    for vector_weight in weights:
        print(f"\n>>> 混合权重 向量={vector_weight:.2f} / BM25={1 - vector_weight:.2f}")
        retriever.hybrid_vector_weight = vector_weight
        results: list[dict] = []
        for question in questions:
            query_vector = encode_query(embedder, question.query)
            hits = retriever.retrieve(
                RecallRoute.HYBRID,
                question.query,
                query_vector,
                top_k,
                stock_code=question.stock_code,
                query_type=question.query_type,
            )
            results.append(evaluate_hits(question, hits, "hybrid"))

        metrics = aggregate_metrics(results)
        rows.append(
            {
                "hybrid_vector_weight": vector_weight,
                "hybrid_bm25_weight": round(1.0 - vector_weight, 2),
                "question_count": int(metrics["question_count"]),
                "recall_at_3": round(metrics["recall_at_3"], 4),
                "recall_at_5": round(metrics["recall_at_5"], 4),
                "recall_at_10": round(metrics["recall_at_10"], 4),
                "mrr": round(metrics["mrr"], 4),
                "hit_rate": round(metrics["hit_rate"], 4),
                "top_k": top_k,
            }
        )
        print(
            f"Recall@10={metrics['recall_at_10']:.1%}  "
            f"MRR={metrics['mrr']:.3f}  "
            f"命中={int(metrics['hit_rate'] * metrics['question_count'])}/"
            f"{int(metrics['question_count'])}"
        )

    retriever.close()
    del embedder
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="混合检索权重分组评测")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=DEFAULT_WEIGHTS,
        help="向量权重列表，如 0.5 0.4 0.3",
    )
    args = parser.parse_args()

    df = run_sweep(args.questions, args.weights, top_k=args.top_k)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "eval_hybrid_weight_sweep.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n对比表已写入：{output_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
