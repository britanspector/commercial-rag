"""
Rerank 评测第二阶段（独立进程，仅加载 Reranker，不加载 Embedding / Milvus）。

用法：
    python src/eval_rerank_phase2.py <pool.pkl> <questions.jsonl> <skip_answer> <refusal_threshold>
    # skip_answer: 1=仅检索对比, 0=含生成/拒答评测
"""

from __future__ import annotations

import pickle
import sys
import traceback
from pathlib import Path

import pandas as pd
from tqdm import tqdm

CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

PROJECT_ROOT = CURRENT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "eval"

from embed_chunks import resolve_device
from eval_retrieval import load_questions
from eval_rerank_common import (
    FINAL_TOP_K,
    aggregate_answer_metrics,
    aggregate_retrieval_metrics,
    evaluate_answer_row,
    evaluate_strategy_hits,
)
from rag_constants import REFUSAL_MESSAGE
from reranker import BGEReranker


def save_report(
    baseline_rows: list[dict],
    rerank_rows: list[dict],
    baseline_answer_rows: list[dict],
    rerank_answer_rows: list[dict],
    skip_answer: bool,
) -> None:
    baseline_metrics = aggregate_retrieval_metrics(baseline_rows)
    rerank_metrics = aggregate_retrieval_metrics(rerank_rows)

    comparison = [
        {
            "strategy": "hybrid_direct_top5",
            "strategy_label": f"混合直接 Top{FINAL_TOP_K}",
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in baseline_metrics.items()},
        },
        {
            "strategy": "hybrid_top20_rerank_top5",
            "strategy_label": f"混合 Top20 → Rerank → Top{FINAL_TOP_K}",
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in rerank_metrics.items()},
        },
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = OUTPUT_DIR / "eval_rerank_comparison.csv"
    pd.DataFrame(comparison).to_csv(comparison_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(baseline_rows + rerank_rows).to_csv(
        OUTPUT_DIR / "eval_rerank_results.csv", index=False, encoding="utf-8-sig"
    )

    print("\n" + "=" * 70)
    print("Rerank 检索对比")
    print(pd.DataFrame(comparison).to_string(index=False))
    print(f"\n对比表：{comparison_path}")

    delta_recall = rerank_metrics["recall_at_5"] - baseline_metrics["recall_at_5"]
    delta_top1 = rerank_metrics["top1_accuracy"] - baseline_metrics["top1_accuracy"]
    delta_mrr = rerank_metrics["mrr"] - baseline_metrics["mrr"]
    print(f"\nΔ Recall@5：  {delta_recall:+.1%}")
    print(f"Δ Top-1 准确率：{delta_top1:+.1%}")
    print(f"Δ MRR：       {delta_mrr:+.3f}")

    if not skip_answer:
        baseline_answer_metrics = aggregate_answer_metrics(baseline_answer_rows)
        rerank_answer_metrics = aggregate_answer_metrics(rerank_answer_rows)
        answer_comparison = [
            {"strategy": "hybrid_direct_top5", **baseline_answer_metrics},
            {"strategy": "hybrid_top20_rerank_top5", **rerank_answer_metrics},
        ]
        answer_path = OUTPUT_DIR / "eval_rerank_answer_comparison.csv"
        pd.DataFrame(answer_comparison).to_csv(
            answer_path, index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(baseline_answer_rows + rerank_answer_rows).to_csv(
            OUTPUT_DIR / "eval_rerank_answer_results.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print("\n" + "=" * 70)
        print("生成答案对比（引用溯源 + 拒答）")
        print(pd.DataFrame(answer_comparison).to_string(index=False))
        print(f"答案对比表：{answer_path}")
        print(f"拒答文案：{REFUSAL_MESSAGE}")
    print("=" * 70)


def main() -> None:
    if len(sys.argv) < 5:
        print(
            "用法: python src/eval_rerank_phase2.py "
            "<pool.pkl> <questions.jsonl> <skip_answer:0|1> <refusal_threshold>",
            file=sys.stderr,
        )
        sys.exit(2)

    pool_cache = Path(sys.argv[1]).resolve()
    questions_path = Path(sys.argv[2]).resolve()
    skip_answer = sys.argv[3].strip() == "1"
    refusal_threshold = float(sys.argv[4])

    if not pool_cache.is_file():
        raise FileNotFoundError(f"未找到召回缓存：{pool_cache}")
    if not questions_path.is_file():
        raise FileNotFoundError(f"未找到评测集：{questions_path}")

    questions = load_questions(questions_path)
    with open(pool_cache, "rb") as cache_file:
        pools: list[list[dict]] = pickle.load(cache_file)

    if len(questions) != len(pools):
        raise ValueError(
            f"评测题数量（{len(questions)}）与召回缓存条数（{len(pools)}）不一致，"
            "请重新运行 python src/eval_rerank.py 生成混合 pool 缓存。"
        )

    print("[2/2] Rerank 与指标统计（独立进程）")
    print(f"  评测题：{len(questions)}，skip_answer={skip_answer}，阈值={refusal_threshold}")

    try:
        reranker = BGEReranker(device=resolve_device())
    except Exception:
        print("\n[错误] Reranker 加载失败。常见原因：", file=sys.stderr)
        print("  - 系统内存 / 页面文件不足（Windows os error 1455）", file=sys.stderr)
        print(
            "  - 模型未下载：python -c \"from huggingface_hub import snapshot_download; "
            "snapshot_download('BAAI/bge-reranker-v2-m3')\"",
            file=sys.stderr,
        )
        traceback.print_exc()
        sys.exit(1)

    baseline_rows: list[dict] = []
    rerank_rows: list[dict] = []
    baseline_answer_rows: list[dict] = []
    rerank_answer_rows: list[dict] = []

    for question, pool_hits in tqdm(
        zip(questions, pools),
        total=len(questions),
        desc="Rerank 评测",
    ):
        direct_hits = pool_hits[:FINAL_TOP_K]
        reranked_hits = reranker.rerank_hits(
            question.query, pool_hits, top_k=FINAL_TOP_K, normalize=True
        )

        baseline_rows.append(
            evaluate_strategy_hits(question, direct_hits, "hybrid_direct_top5")
        )
        rerank_rows.append(
            evaluate_strategy_hits(question, reranked_hits, "hybrid_top20_rerank_top5")
        )

        if not skip_answer:
            direct_scored = reranker.rerank_hits(
                question.query, direct_hits, top_k=len(direct_hits), normalize=True
            )
            baseline_answer_rows.append(
                evaluate_answer_row(
                    question,
                    direct_scored,
                    "hybrid_direct_top5",
                    refusal_threshold,
                    reranker=None,
                )
            )
            rerank_answer_rows.append(
                evaluate_answer_row(
                    question,
                    reranked_hits,
                    "hybrid_top20_rerank_top5",
                    refusal_threshold,
                    reranker=None,
                )
            )

    del reranker
    save_report(
        baseline_rows,
        rerank_rows,
        baseline_answer_rows,
        rerank_answer_rows,
        skip_answer,
    )


if __name__ == "__main__":
    main()
