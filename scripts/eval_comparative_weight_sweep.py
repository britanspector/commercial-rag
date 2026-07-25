"""
comparative 专项 hybrid 权重扫描。

默认比较三种 retrieval 变体：
- hybrid_plain
- hybrid_rrf_fallback
- hybrid_subq_indep
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval_comparative_ablation import DEFAULT_QUESTIONS, load_questions, run_variant_eval
from eval_retrieval import load_chunk_id_set, validate_questions


def main() -> None:
    parser = argparse.ArgumentParser(description="comparative hybrid 权重扫描")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["hybrid_plain", "hybrid_rrf_fallback", "hybrid_subq_indep"],
        choices=["hybrid_plain", "hybrid_rrf_fallback", "hybrid_subq_shared", "hybrid_subq_indep"],
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hybrid-pool-size", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "eval" / "comparative_100" / "comparative_weight_sweep.csv",
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)
    validate_questions([item.eval_question for item in questions], load_chunk_id_set())

    rows: list[dict] = []
    for variant in args.variants:
        for weight in args.weights:
            detail_rows, metrics, _ = run_variant_eval(
                questions,
                variant=variant,
                top_k=args.top_k,
                hybrid_vector_weight=weight,
                hybrid_pool_size=args.hybrid_pool_size,
            )
            rows.append(
                {
                    "variant": variant,
                    "question_count": int(metrics["question_count"]),
                    "hybrid_vector_weight": weight,
                    "recall_at_3": round(metrics["recall_at_3"], 4),
                    "recall_at_5": round(metrics["recall_at_5"], 4),
                    "recall_at_10": round(metrics["recall_at_10"], 4),
                    "mrr": round(metrics["mrr"], 4),
                    "hit_rate": round(metrics["hit_rate"], 4),
                    "multi_company_top5_rate": round(metrics["multi_company_top5_rate"], 4),
                    "comparative_insufficient_rate": round(metrics["comparative_insufficient_rate"], 4),
                    "numeric_like_recall_at_10": round(
                        sum(
                            1
                            for row in detail_rows
                            if row["compare_tag"]
                            in {
                                "numeric_revenue",
                                "numeric_profit",
                                "numeric_growth",
                                "numeric_margin",
                                "numeric_scale",
                                "numeric_forecast",
                                "numeric_eps",
                                "numeric_generation",
                                "numeric_quality",
                            }
                            and row["hit"]
                        )
                        / max(
                            1,
                            sum(
                                1
                                for row in detail_rows
                                if row["compare_tag"]
                                in {
                                    "numeric_revenue",
                                    "numeric_profit",
                                    "numeric_growth",
                                    "numeric_margin",
                                    "numeric_scale",
                                    "numeric_forecast",
                                    "numeric_eps",
                                    "numeric_generation",
                                    "numeric_quality",
                                }
                            ),
                        ),
                        4,
                    ),
                    "logic_like_recall_at_10": round(
                        sum(
                            1
                            for row in detail_rows
                            if row["compare_tag"]
                            in {
                                "logic_compare",
                                "growth_driver",
                                "structure_compare",
                                "structure_positioning",
                                "industry_view",
                                "risk_compare",
                                "same_company_multi_report",
                                "valuation_rating",
                                "rating_compare",
                                "mixed_financial",
                            }
                            and row["hit"]
                        )
                        / max(
                            1,
                            sum(
                                1
                                for row in detail_rows
                                if row["compare_tag"]
                                in {
                                    "logic_compare",
                                    "growth_driver",
                                    "structure_compare",
                                    "structure_positioning",
                                    "industry_view",
                                    "risk_compare",
                                    "same_company_multi_report",
                                    "valuation_rating",
                                    "rating_compare",
                                    "mixed_financial",
                                }
                            ),
                        ),
                        4,
                    ),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n输出文件：{args.output}")


if __name__ == "__main__":
    main()
