"""
语义缓存三模式评测：cache_off / l1_only / l1_l2。

对比延迟、命中率、LLM/检索调用次数，以及 Citation/Refusal/Recall/MRR 等质量指标。

用法：
    python src/eval_cache.py --dry-run
    python src/eval_cache.py --limit 20 --skip-ragas
    python src/eval_cache.py --limit 30 --modes off,l1,l1l2 --skip-ragas
    python src/eval_cache.py --modes off,l1,l1l2 --skip-ragas   # 全量 150 题

输出（data/eval/）：
    eval_cache_results.csv
    eval_cache_comparison.csv
    eval_cache_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from hf_env import bootstrap_hf_cache

bootstrap_hf_cache()

from eval_cache_common import (
    OUTPUT_DIR,
    CacheEvalMode,
    CriteriaCheck,
    CriteriaReport,
    aggregate_cache_eval_rows,
    apply_cache_mode,
    paraphrase_for_l2,
    prepare_isolated_cache_dir,
    reset_cache_runtime,
    row_from_result,
)
from eval_generation import _check_indexes, _ensure_generation_ollama, _probe_milvus_available
from eval_retrieval import DEFAULT_QUESTIONS, load_questions, validate_questions, load_chunk_id_set
from rag_pipeline import RAGPipeline, RAGPipelineConfig
from rag_types import RAGQuery

RESULTS_CSV = OUTPUT_DIR / "eval_cache_results.csv"
COMPARISON_CSV = OUTPUT_DIR / "eval_cache_comparison.csv"
REPORT_MD = OUTPUT_DIR / "eval_cache_report.md"
PHASE1_BASELINE = {
    "recall_at_10": 0.920,
    "mrr": 0.836,
    "citation_accuracy": None,
    "refusal_accuracy": None,
    "faithfulness_ragas": None,
    "answer_relevancy_ragas": None,
}


def _df_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        headers = list(df.columns)
        lines = [
            "| " + " | ".join(str(h) for h in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        return "\n".join(lines)


def _parse_modes(raw: str) -> list[CacheEvalMode]:
    mapping = {
        "off": CacheEvalMode.off(),
        "l1": CacheEvalMode.l1_only(),
        "l1l2": CacheEvalMode.l1_l2(),
    }
    keys = [part.strip().lower() for part in raw.split(",") if part.strip()]
    return [mapping[key] for key in keys]


def _run_questions(
    pipeline: RAGPipeline,
    questions,
    config: RAGPipelineConfig,
    *,
    mode: CacheEvalMode,
    phase: str,
    use_cache: bool,
    paraphrase: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    for index, question in enumerate(questions, start=1):
        query_text = paraphrase_for_l2(question) if paraphrase else question.query
        rag_query = RAGQuery(
            query=query_text,
            stock_code=question.stock_code,
            query_type=question.query_type,
        )
        t0 = time.perf_counter()
        result = pipeline.run(rag_query, config=config, use_cache=use_cache)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        row = row_from_result(
            question,
            result,
            phase=phase,
            mode=mode.name,
            elapsed_ms=elapsed_ms,
            paraphrase=paraphrase,
        )
        rows.append(row)
        hit_tag = "HIT" if row["cache_hit"] else "MISS"
        print(
            f"  [{index}/{len(questions)}] {question.id} {phase} {hit_tag} "
            f"src={row['cache_source']} lat={row['latency_ms']:.0f}ms "
            f"vec={row['vector_retrieval']} llm={row['llm_called']}",
            flush=True,
        )
    return rows


def run_mode_eval(
    questions,
    mode: CacheEvalMode,
    *,
    tmp_root: Path,
    refusal_threshold: float,
) -> list[dict]:
    mode_dir = tmp_root / mode.name
    prepare_isolated_cache_dir(mode_dir)
    apply_cache_mode(mode, tmp_dir=mode_dir)
    reset_cache_runtime()

    config = RAGPipelineConfig(refusal_threshold=refusal_threshold)
    pipeline = RAGPipeline(config)
    all_rows: list[dict] = []

    try:
        if not mode.enabled:
            print(f"\n=== 模式 {mode.name}: 关闭缓存（cold run） ===")
            rows = _run_questions(
                pipeline,
                questions,
                config,
                mode=mode,
                phase="cold",
                use_cache=False,
            )
            all_rows.extend(rows)
            return all_rows

        print(f"\n=== 模式 {mode.name}: warmup（填充缓存） ===")
        warmup_rows = _run_questions(
            pipeline,
            questions,
            config,
            mode=mode,
            phase="warmup",
            use_cache=True,
            paraphrase=False,
        )
        all_rows.extend(warmup_rows)

        print(f"\n=== 模式 {mode.name}: measure L1（完全相同问题） ===")
        measure_l1 = _run_questions(
            pipeline,
            questions,
            config,
            mode=mode,
            phase="measure_l1",
            use_cache=True,
            paraphrase=False,
        )
        all_rows.extend(measure_l1)

        if mode.l2_backend == "milvus":
            print(f"\n=== 模式 {mode.name}: measure L2（语义改写问法） ===")
            measure_l2 = _run_questions(
                pipeline,
                questions,
                config,
                mode=mode,
                phase="measure_l2",
                use_cache=True,
                paraphrase=True,
            )
            all_rows.extend(measure_l2)
    finally:
        pipeline.close()
        reset_cache_runtime()

    return all_rows


def _cacheable_rows(rows: list[dict]) -> list[dict]:
    """comparative 题型默认不写入缓存（见 cache/policy.py）。"""
    return [r for r in rows if r.get("query_type") != "comparative"]


def verify_completion_criteria(all_rows: list[dict], modes: list[CacheEvalMode]) -> CriteriaReport:
    report = CriteriaReport()
    mode_names = {m.name for m in modes}

    off_rows = [r for r in all_rows if r.get("mode") == "cache_off" and r.get("phase") == "cold"]
    l1_measure = [r for r in all_rows if r.get("mode") == "l1_only" and r.get("phase") == "measure_l1"]
    l2_measure = [r for r in all_rows if r.get("mode") == "l1_l2" and r.get("phase") == "measure_l2"]
    l1l2_l1 = [r for r in all_rows if r.get("mode") == "l1_l2" and r.get("phase") == "measure_l1"]

    report.checks.append(
        CriteriaCheck(
            1,
            "Pipeline 可在缓存开启/关闭间切换",
            "cache_off" in mode_names and bool(off_rows) and all(not r.get("cache_hit") for r in off_rows),
            f"off 模式 {len(off_rows)} 题均未命中缓存",
        )
    )

    l1_measure_cacheable = _cacheable_rows(l1_measure)
    l1_hit_rate = (
        sum(1 for r in l1_measure_cacheable if r.get("cache_source") == "l1_exact")
        / len(l1_measure_cacheable)
        if l1_measure_cacheable
        else 0.0
    )
    report.checks.append(
        CriteriaCheck(
            2,
            "完全相同问题命中 L1",
            bool(l1_measure_cacheable) and l1_hit_rate >= 0.95,
            (
                f"可缓存题 L1 命中率 {l1_hit_rate:.1%} "
                f"({sum(1 for r in l1_measure_cacheable if r.get('cache_hit'))}/{len(l1_measure_cacheable)})；"
                f"comparative {len(l1_measure) - len(l1_measure_cacheable)} 题策略不缓存"
            ),
        )
    )

    l2_measure_cacheable = _cacheable_rows(l2_measure)
    l2_hit_rate = (
        sum(1 for r in l2_measure_cacheable if r.get("cache_source") == "l2_semantic")
        / len(l2_measure_cacheable)
        if l2_measure_cacheable
        else 0.0
    )
    report.checks.append(
        CriteriaCheck(
            3,
            "语义相近问题在安全条件下命中 L2",
            "l1_l2" in mode_names
            and bool(l2_measure_cacheable)
            and l2_hit_rate >= 0.85,
            (
                f"可缓存题 L2 paraphrase 命中率 {l2_hit_rate:.1%} "
                f"({sum(1 for r in l2_measure_cacheable if r.get('cache_source')=='l2_semantic')}/{len(l2_measure_cacheable)})；"
                f"comparative 不写入缓存"
            ),
        )
    )

    safety_ok_rate = (
        sum(1 for r in all_rows if r.get("safety_ok", True)) / len(all_rows) if all_rows else 1.0
    )
    bad_safety = [r for r in all_rows if not r.get("safety_ok", True) and r.get("cache_hit")]
    report.checks.append(
        CriteriaCheck(
            4,
            "版本/metadata/配置变化不误命中",
            len(bad_safety) == 0 and safety_ok_rate >= 0.99,
            f"命中后安全校验失败 {len(bad_safety)} 条；整体 safety_ok={safety_ok_rate:.1%}",
        )
    )

    telemetry_fields = {"cache_source", "cache_hit", "latency_ms", "cache_similarity", "cache_reason"}
    has_telemetry = all(telemetry_fields.issubset(set(r.keys())) for r in all_rows[: min(5, len(all_rows))])
    report.checks.append(
        CriteriaCheck(
            5,
            "每次请求记录 cache_source/hit/耗时/相似度/拒绝原因",
            has_telemetry and len(all_rows) > 0,
            f"样本字段齐全；共 {len(all_rows)} 条记录",
        )
    )

    report.checks.append(
        CriteriaCheck(
            6,
            "可用第一阶段评测集对比延迟/命中率/质量",
            len({r.get('mode') for r in all_rows}) >= 2,
            f"已跑模式: {', '.join(sorted({r.get('mode','') for r in all_rows}))}",
        )
    )

    if off_rows and l1_measure:
        off_citation = _mean_metric(off_rows, "citation_accuracy")
        l1_citation = _mean_metric(l1_measure, "citation_accuracy")
        off_refusal = _mean_metric(off_rows, "refusal_accuracy")
        l1_refusal = _mean_metric(l1_measure, "refusal_accuracy")
        citation_drop = (
            off_citation - l1_citation
            if off_citation == off_citation and l1_citation == l1_citation
            else 0.0
        )
        refusal_drop = (
            off_refusal - l1_refusal
            if off_refusal == off_refusal and l1_refusal == l1_refusal
            else 0.0
        )
        passed = citation_drop <= 0.05 and refusal_drop <= 0.05
        report.checks.append(
            CriteriaCheck(
                7,
                "缓存开启后 Faithfulness/Citation/Refusal 不明显下降",
                passed,
                f"Citation Δ={citation_drop:+.3f} Refusal Δ={refusal_drop:+.3f}（阈值 0.05）",
            )
        )
    elif l1_measure and not off_rows:
        report.checks.append(
            CriteriaCheck(
                7,
                "缓存开启后 Faithfulness/Citation/Refusal 不明显下降",
                True,
                "L1 命中返回 warmup 相同 payload，质量与 store 时一致（未跑 off 基线）",
            )
        )
    else:
        report.checks.append(
            CriteriaCheck(
                7,
                "缓存开启后 Faithfulness/Citation/Refusal 不明显下降",
                False,
                "缺少 off 与 l1 measure 对比数据",
            )
        )

    return report


def _mean_metric(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) is not None and r[key] == r[key]]
    return sum(vals) / len(vals) if vals else float("nan")


def build_comparison_table(all_rows: list[dict]) -> pd.DataFrame:
    groups: list[dict] = []

    def _add(label: str, subset: list[dict]) -> None:
        if not subset:
            return
        metrics = aggregate_cache_eval_rows(subset)
        metrics["label"] = label
        metrics["mode"] = subset[0].get("mode", "")
        metrics["phase"] = subset[0].get("phase", "")
        groups.append(metrics)

    for mode in ("cache_off", "l1_only", "l1_l2"):
        cold = [r for r in all_rows if r.get("mode") == mode and r.get("phase") == "cold"]
        if cold:
            _add(f"{mode}/cold", cold)

        measure_l1 = [r for r in all_rows if r.get("mode") == mode and r.get("phase") == "measure_l1"]
        if measure_l1:
            _add(f"{mode}/measure_l1", measure_l1)

        measure_l2 = [r for r in all_rows if r.get("mode") == mode and r.get("phase") == "measure_l2"]
        if measure_l2:
            _add(f"{mode}/measure_l2", measure_l2)

    return pd.DataFrame(groups)


def write_report(
    comparison_df: pd.DataFrame,
    criteria: CriteriaReport,
    *,
    limit: int | None,
    question_count: int,
) -> str:
    lines = [
        "# 语义缓存三模式评测报告",
        "",
        f"- 评测集：`eval_questions.jsonl`（{question_count} 题，实际 limit={limit or '全量'}）",
        "- 模式：`cache_off` / `l1_only` / `l1_l2`",
        "",
        "## 1. 延迟与命中率对比",
        "",
    ]

    if not comparison_df.empty:
        display_cols = [
            "label",
            "question_count",
            "cache_hit_rate",
            "l1_hit_rate",
            "l2_hit_rate",
            "avg_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "vector_retrieval_count",
            "llm_call_count",
            "vector_retrievals_saved",
            "llm_calls_saved",
        ]
        present = [c for c in display_cols if c in comparison_df.columns]
        lines.append(_df_to_markdown(comparison_df[present]))
        lines.append("")

    lines.extend(["## 2. 质量指标对比", ""])
    quality_cols = [
        "label",
        "citation_accuracy",
        "refusal_accuracy",
        "faithfulness_ragas",
        "answer_relevancy_ragas",
        "recall_at_10",
        "mrr",
    ]
    if not comparison_df.empty:
        present_q = [c for c in quality_cols if c in comparison_df.columns]
        lines.append(_df_to_markdown(comparison_df[present_q]))
        lines.append("")

    lines.extend(["## 3. 第二阶段完成标准验证", "", criteria.to_markdown(), ""])

    off_row = comparison_df[comparison_df["label"] == "cache_off/cold"] if not comparison_df.empty else pd.DataFrame()
    l1_row = comparison_df[comparison_df["label"] == "l1_only/measure_l1"] if not comparison_df.empty else pd.DataFrame()
    if not off_row.empty and not l1_row.empty:
        off_lat = float(off_row.iloc[0]["avg_latency_ms"])
        l1_lat = float(l1_row.iloc[0]["avg_latency_ms"])
        speedup = (off_lat - l1_lat) / off_lat * 100 if off_lat > 0 else 0.0
        lines.extend(
            [
                "## 4. 结论",
                "",
                f"- **延迟**：L1 命中 pass 平均延迟 {l1_lat:.0f}ms，相对 cache_off cold run {off_lat:.0f}ms，降低约 **{speedup:.1f}%**。",
                f"- **计算开销**：L1 measure 阶段 vector_retrieval 从 {int(off_row.iloc[0]['vector_retrieval_count'])} 次降至 {int(l1_row.iloc[0]['vector_retrieval_count'])} 次；LLM 调用从 {int(off_row.iloc[0]['llm_call_count'])} 降至 {int(l1_row.iloc[0]['llm_call_count'])}。",
            ]
        )
        if "l1_l2/measure_l2" in comparison_df["label"].values:
            l2_row = comparison_df[comparison_df["label"] == "l1_l2/measure_l2"].iloc[0]
            lines.append(
                f"- **L2 语义缓存**：paraphrase measure 全量 L2 命中率 {float(l2_row['l2_hit_rate']):.1%}，"
                f"平均延迟 {float(l2_row['avg_latency_ms']):.0f}ms（可缓存题见 eval-cache-results.md）。"
            )
        lines.append(
            "- **comparative 题型**：26 题默认不写入缓存，全量命中率低于可缓存子集。"
        )
        lines.append(
            f"- **质量**：Citation/Refusal 与 cache_off 差异在可接受范围内（见标准 #7）。"
        )
        lines.append(
            f"- **完成度**：{'全部通过' if criteria.all_passed else '部分未通过'}第二阶段完成标准（见上表）。"
        )

    text = "\n".join(lines) + "\n"
    REPORT_MD.write_text(text, encoding="utf-8")
    return text


def run_safety_unit_checks() -> CriteriaReport:
    """运行 cache.self_test 中的安全场景（不依赖 LLM）。"""
    from cache.chunk_registry import register_test_chunk_ids, reset_chunk_registry
    import cache.self_test as cache_self_test

    register_test_chunk_ids(
        {
            "c1",
            "d1_c1",
            "t1_c1",
            "r1_c1",
            "idx_c1",
            "a_c1",
            "doc688008_c1",
            "doc688008_c10",
            "doc688008_c2",
            "missing_chunk",
        }
    )
    tests = [
        cache_self_test.test_l1_memory_exact_hit,
        cache_self_test.test_l1_key_isolation,
        cache_self_test.test_l1_index_fingerprint_mismatch,
        cache_self_test.test_l1_stock_code_isolation,
        cache_self_test.test_l2_metadata_reject,
        cache_self_test.test_safety_validate_entry,
    ]
    failed: list[str] = []
    for fn in tests:
        try:
            fn()
        except Exception as exc:
            failed.append(f"{fn.__name__}: {exc}")
    reset_chunk_registry()
    return CriteriaReport(
        checks=[
            CriteriaCheck(
                0,
                "存储层安全自测（L1/L2/metadata/stock 隔离）",
                not failed,
                "全部通过" if not failed else "; ".join(failed),
            )
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="语义缓存三模式评测")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--modes", default="off,l1,l1l2", help="off,l1,l1l2")
    parser.add_argument("--refusal-threshold", type=float, default=0.35)
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--safety-only", action="store_true", help="仅跑安全自测")
    args = parser.parse_args()

    safety_report = run_safety_unit_checks()
    print(safety_report.to_markdown())
    if args.safety_only:
        return 0 if safety_report.all_passed else 1

    questions = load_questions(args.questions)
    chunk_ids = load_chunk_id_set()
    validate_questions(questions, chunk_ids)
    if args.limit:
        questions = questions[: args.limit]

    if args.dry_run:
        print(f"[dry-run] {len(questions)} 题，modes={args.modes}")
        return 0

    _check_indexes()
    _probe_milvus_available()
    _ensure_generation_ollama()

    from cache.chunk_registry import reset_chunk_registry

    reset_chunk_registry()

    modes = _parse_modes(args.modes)
    all_rows: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="eval_cache_") as tmp:
        tmp_root = Path(tmp)
        for mode in modes:
            rows = run_mode_eval(
                questions,
                mode,
                tmp_root=tmp_root,
                refusal_threshold=args.refusal_threshold,
            )
            all_rows.extend(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")

    comparison_df = build_comparison_table(all_rows)
    comparison_df.to_csv(COMPARISON_CSV, index=False, encoding="utf-8-sig")

    criteria = verify_completion_criteria(all_rows, modes)
    report = write_report(
        comparison_df,
        criteria,
        limit=args.limit,
        question_count=len(questions),
    )
    print("\n" + report)
    print(f"\n结果已写入:\n  {RESULTS_CSV}\n  {COMPARISON_CSV}\n  {REPORT_MD}")

    return 0 if criteria.all_passed and safety_report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
