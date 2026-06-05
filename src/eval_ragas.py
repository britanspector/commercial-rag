"""
在已有 Pipeline 评测结果上补跑 RAGAS（不重跑 Milvus / Rerank）。

默认：Ollama qwen3:8b + 本地 bge-large-zh Embedding。

用法：
    ollama pull qwen3:8b
    ollama serve

    # 基于 detail.jsonl（需先 eval_generation.py --save-detail）
    python src/eval_ragas.py
    python src/eval_ragas.py --limit 5

    # 调试单题 / 单指标
    python src/eval_ragas.py --question-id q02
    python src/eval_ragas.py --question-id q02 --metrics faith

    # 仅补跑尚未有 RAGAS 分的题
    python src/eval_ragas.py --resume

环境变量见 src/eval_ragas_config.py（RAGAS_BACKEND、RAGAS_LLM_MODEL 等）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from eval_generation_common import aggregate_by_query_type, aggregate_generation_metrics
from eval_ragas_runner import (
    check_langchain_stack,
    ensure_stdout_unbuffered,
    filter_rows_for_eval,
    load_rows_for_ragas,
    run_ragas_on_rows,
)

PROJECT_ROOT = CURRENT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "eval"
DETAIL_JSONL = OUTPUT_DIR / "eval_generation_detail.jsonl"
RESULTS_CSV = OUTPUT_DIR / "eval_generation_results.csv"
METRICS_CSV = OUTPUT_DIR / "eval_generation_metrics.csv"
METRICS_BY_TYPE_CSV = OUTPUT_DIR / "eval_generation_metrics_by_query_type.csv"
RAGAS_DETAIL_JSONL = OUTPUT_DIR / "eval_generation_detail_ragas.jsonl"


def _rows_need_ragas(row: dict) -> bool:
    if row.get("refused"):
        return False
    if row.get("ragas_skipped"):
        return False
    faith = row.get("ragas_faithfulness")
    if faith is not None and faith == faith:
        return False
    return bool(row.get("ragas_contexts"))


def save_ragas_reports(rows: list[dict], detail_out: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    flat_rows = []
    for row in rows:
        flat = dict(row)
        for json_key in ("citations_json", "evidence_check_json", "ragas_contexts"):
            if json_key in flat and not isinstance(flat[json_key], str):
                flat[json_key] = json.dumps(flat[json_key], ensure_ascii=False)
        flat_rows.append(flat)

    pd.DataFrame(flat_rows).to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    metrics = aggregate_generation_metrics(rows)
    pd.DataFrame([metrics]).to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
    by_type = aggregate_by_query_type(rows)
    pd.DataFrame(by_type).to_csv(METRICS_BY_TYPE_CSV, index=False, encoding="utf-8-sig")

    with open(detail_out, "w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n已写入：\n  {RESULTS_CSV}\n  {METRICS_CSV}\n  {METRICS_BY_TYPE_CSV}\n  {detail_out}")
    print("\n=== RAGAS 整体指标 ===")
    for key in (
        "faithfulness_ragas",
        "answer_relevancy_ragas",
        "ragas_scored_n",
        "refusal_accuracy",
        "refusal_accuracy_retrieval",
        "citation_accuracy",
    ):
        value = metrics.get(key)
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}" if value == value else f"  {key}: nan")
        else:
            print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 RAGAS 补跑（Ollama Qwen3-8B）")
    parser.add_argument("--detail", type=Path, default=DETAIL_JSONL)
    parser.add_argument("--results", type=Path, default=RESULTS_CSV)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="跳过已有 ragas_faithfulness 的题")
    parser.add_argument("--ragas-backend", type=str, default=None, help="ollama|openai|auto")
    parser.add_argument("--ragas-llm", type=str, default=None, help="如 qwen3:8b")
    parser.add_argument("--question-id", type=str, default=None, help="仅评测指定题号，如 q04")
    parser.add_argument("--index", type=int, default=None, help="仅评测第 N 题（1-based）")
    parser.add_argument(
        "--metrics",
        type=str,
        default="all",
        choices=["all", "faith", "rel", "faithfulness", "answer_relevancy"],
        help="仅跑指定指标（调试用）",
    )
    parser.add_argument(
        "--output-detail",
        type=Path,
        default=RAGAS_DETAIL_JSONL,
        help="带 RAGAS 分的 detail 输出路径",
    )
    args = parser.parse_args()

    ensure_stdout_unbuffered()
    check_langchain_stack()

    rows = load_rows_for_ragas(args.detail.resolve(), args.results.resolve())
    if args.question_id or args.index is not None:
        rows = filter_rows_for_eval(rows, question_id=args.question_id, index=args.index)
        print(f"[debug] 单条评测：{rows[0].get('question_id')}")
    elif args.limit is not None:
        rows = rows[: args.limit]

    if args.resume:
        pending = [r for r in rows if _rows_need_ragas(r)]
        print(f"[resume] 待打分 {len(pending)} / {len(rows)} 题")
        if not pending:
            print("全部已有 RAGAS 分，无需补跑。")
            save_ragas_reports(rows, args.output_detail.resolve())
            return
        _, rows = run_ragas_on_rows(
            rows,
            backend=args.ragas_backend,
            llm_model=args.ragas_llm,
            metrics=args.metrics,
        )
    else:
        to_score = [r for r in rows if not r.get("refused")]
        print(f"将对 {len(to_score)} 道非拒答题运行 RAGAS（共 {len(rows)} 题）")
        _, rows = run_ragas_on_rows(
            rows,
            backend=args.ragas_backend,
            llm_model=args.ragas_llm,
            metrics=args.metrics,
        )

    save_ragas_reports(rows, args.output_detail.resolve())


if __name__ == "__main__":
    main()
