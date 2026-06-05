"""
评测任务入口：供 CLI 与 POST /eval 复用。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

PROJECT_ROOT = CURRENT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "eval"


def run_generation_eval_job(
    *,
    limit: int | None = None,
    skip_ragas: bool = True,
    save_detail: bool = True,
    resume: bool = False,
    refusal_threshold: float | None = None,
) -> dict[str, Any]:
    from eval_generation import (
        DETAIL_JSONL,
        METRICS_BY_TYPE_CSV,
        METRICS_CSV,
        RESULTS_CSV,
        run_generation_eval,
        save_reports,
    )
    from eval_retrieval import DEFAULT_QUESTIONS, load_chunk_id_set, load_questions, validate_questions
    from rag_constants import DEFAULT_RERANK_REFUSAL_THRESHOLD

    questions = load_questions(DEFAULT_QUESTIONS)
    validate_questions(questions, load_chunk_id_set())
    threshold = (
        refusal_threshold if refusal_threshold is not None else DEFAULT_RERANK_REFUSAL_THRESHOLD
    )
    rows = run_generation_eval(
        questions,
        refusal_threshold=threshold,
        limit=limit,
        skip_ragas=skip_ragas,
        save_detail=save_detail,
        resume=resume,
    )
    save_reports(rows, save_detail=save_detail)
    from eval_generation_common import aggregate_generation_metrics

    metrics = aggregate_generation_metrics(rows)
    return {
        "metrics": metrics,
        "outputs": {
            "results_csv": str(RESULTS_CSV),
            "metrics_csv": str(METRICS_CSV),
            "metrics_by_type_csv": str(METRICS_BY_TYPE_CSV),
            "detail_jsonl": str(DETAIL_JSONL) if save_detail else "",
        },
        "question_count": len(rows),
    }


def run_retrieval_eval_job(
    *,
    compare_routes: bool = False,
    route: str = "hybrid",
    top_k: int = 10,
    legacy_retriever: bool = False,
    pipeline_stage: str = "rerank",
) -> dict[str, Any]:
    from eval_retrieval import (
        OUTPUT_DIR,
        load_chunk_id_set,
        load_questions,
        run_compare_routes,
        run_retrieval_eval,
        save_reports,
        validate_questions,
        DEFAULT_QUESTIONS,
    )
    from retrieval import DEFAULT_HYBRID_VECTOR_WEIGHT

    questions = load_questions(DEFAULT_QUESTIONS)
    validate_questions(questions, load_chunk_id_set())
    use_pipeline = not legacy_retriever

    if compare_routes:
        run_compare_routes(
            questions,
            top_k,
            DEFAULT_HYBRID_VECTOR_WEIGHT,
            use_pipeline_hybrid=use_pipeline,
            pipeline_stage=pipeline_stage,
        )
        return {
            "mode": "compare_routes",
            "outputs": {
                "comparison_csv": str(OUTPUT_DIR / "eval_route_comparison.csv"),
            },
        }

    results, metrics = run_retrieval_eval(
        questions,
        route,
        top_k,
        DEFAULT_HYBRID_VECTOR_WEIGHT,
        use_pipeline=use_pipeline and route == "hybrid",
        pipeline_stage=pipeline_stage,
    )
    out_route = "hybrid_pipeline" if use_pipeline and route == "hybrid" else route
    save_reports(results, metrics, out_route, top_k)
    return {
        "mode": "single_route",
        "route": out_route,
        "metrics": metrics,
        "outputs": {
            "results_csv": str(OUTPUT_DIR / f"eval_results_{out_route}.csv"),
            "metrics_csv": str(OUTPUT_DIR / f"eval_metrics_{out_route}.csv"),
        },
    }


def run_ragas_eval_job(*, resume: bool = True, limit: int | None = None) -> dict[str, Any]:
    from eval_generation_common import aggregate_generation_metrics
    from eval_ragas import (
        DETAIL_JSONL,
        RAGAS_DETAIL_JSONL,
        RESULTS_CSV,
        METRICS_CSV,
        _rows_need_ragas,
        save_ragas_reports,
    )
    from eval_ragas_runner import check_langchain_stack, load_rows_for_ragas, run_ragas_on_rows

    check_langchain_stack()
    rows = load_rows_for_ragas(DETAIL_JSONL.resolve(), RESULTS_CSV.resolve())
    if limit is not None:
        rows = rows[:limit]

    if resume:
        pending = [r for r in rows if _rows_need_ragas(r)]
        if pending:
            _, rows = run_ragas_on_rows(rows)
    else:
        _, rows = run_ragas_on_rows(rows)

    save_ragas_reports(rows, RAGAS_DETAIL_JSONL.resolve())
    metrics = aggregate_generation_metrics(rows)
    return {
        "metrics": metrics,
        "outputs": {
            "detail_ragas_jsonl": str(RAGAS_DETAIL_JSONL),
            "metrics_csv": str(METRICS_CSV),
        },
        "ragas_scored_n": metrics.get("ragas_scored_n"),
    }
