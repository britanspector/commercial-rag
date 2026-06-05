"""
生成质量离线评测（150 题）：RAG Pipeline 全链路 + Citation/Refusal + 可选 RAGAS。

前置：
    python src/embed_chunks.py
    python src/build_bm25_index.py

用法：
    python src/eval_generation.py --dry-run
    python src/eval_generation.py --limit 10
    python src/eval_generation.py --skip-ragas --save-detail
    python src/eval_generation.py --resume --skip-ragas

本地 RAGAS（Ollama，不重跑 Pipeline）：
    ollama pull qwen3:8b
    python src/eval_ragas.py

输出（data/eval/）：
    eval_generation_results.csv
    eval_generation_metrics.csv
    eval_generation_metrics_by_query_type.csv
    eval_generation_detail.jsonl      （--save-detail）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from hf_env import bootstrap_hf_cache

bootstrap_hf_cache()

from embed_chunks import OUTPUT_MILVUS_DB
from bm25_store import DEFAULT_INDEX_PATH
from eval_generation_common import (
    aggregate_by_query_type,
    aggregate_generation_metrics,
    row_from_pipeline,
    run_ragas_evaluate,
)
from eval_retrieval import DEFAULT_QUESTIONS, load_chunk_id_set, load_questions, validate_questions
from eval_ragas_runner import ensure_stdout_unbuffered
from generation_config import describe_generation_config, resolve_generation_config
from pipeline.llm_client import check_ollama_reachable, ensure_ollama_model
from rag_constants import DEFAULT_RERANK_REFUSAL_THRESHOLD
from rag_pipeline import RAGPipeline, RAGPipelineConfig
from rag_types import RAGQuery

PROJECT_ROOT = CURRENT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "eval"
RESULTS_CSV = OUTPUT_DIR / "eval_generation_results.csv"
METRICS_CSV = OUTPUT_DIR / "eval_generation_metrics.csv"
METRICS_BY_TYPE_CSV = OUTPUT_DIR / "eval_generation_metrics_by_query_type.csv"
DETAIL_JSONL = OUTPUT_DIR / "eval_generation_detail.jsonl"


def _check_indexes() -> None:
    if not OUTPUT_MILVUS_DB.exists():
        raise FileNotFoundError(f"请先运行 embed_chunks.py：{OUTPUT_MILVUS_DB}")
    if not DEFAULT_INDEX_PATH.exists():
        raise FileNotFoundError(f"请先运行 build_bm25_index.py：{DEFAULT_INDEX_PATH}")


def _probe_milvus_available() -> None:
    """启动前探测 Milvus Lite 是否被其它进程占用。"""
    from pymilvus import MilvusClient

    try:
        client = MilvusClient(uri=str(OUTPUT_MILVUS_DB))
        client.list_collections()
    except Exception as error:
        message = str(error).lower()
        if "lock" in message or "datadir" in message or "already" in message:
            raise RuntimeError(
                f"Milvus 数据库可能被占用：{OUTPUT_MILVUS_DB}\n"
                "请先停止 uvicorn / 其它评测进程后再运行。"
            ) from error
        raise


def _load_resume_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "question_id" not in df.columns:
        return {}
    rows_by_id: dict[str, dict] = {}
    for record in df.to_dict(orient="records"):
        question_id = str(record.get("question_id", ""))
        if question_id:
            rows_by_id[question_id] = record
    return rows_by_id


def _format_refusal_hint(row: dict) -> str:
    if row.get("refusal_correct"):
        return ""
    parts = []
    if row.get("refusal_reason"):
        parts.append(f"reason={row['refusal_reason']}")
    if row.get("refused") and row.get("retrieval_hit"):
        parts.append("高rerank仍拒答(evidence)")
    if not row.get("refused") and row.get("should_refuse"):
        parts.append("未命中仍作答")
    return " " + " ".join(parts) if parts else ""


def _ensure_generation_ollama() -> None:
    cfg = resolve_generation_config()
    print(f"[生成] {describe_generation_config(cfg)}")
    if not check_ollama_reachable(cfg.ollama_base_url):
        raise RuntimeError(
            f"无法连接 Ollama：{cfg.ollama_base_url}。请先执行：ollama serve"
        )
    ensure_ollama_model(cfg)


def run_generation_eval(
    questions,
    *,
    refusal_threshold: float,
    limit: int | None = None,
    skip_ragas: bool = False,
    ragas_llm: str | None = None,
    ragas_backend: str | None = None,
    save_detail: bool = False,
    resume: bool = False,
) -> list[dict]:
    _check_indexes()
    _probe_milvus_available()
    _ensure_generation_ollama()

    if limit is not None:
        questions = questions[:limit]

    existing_by_id = _load_resume_rows(RESULTS_CSV) if resume else {}
    if resume and existing_by_id:
        print(f"[resume] 已有 {len(existing_by_id)} 题结果，将跳过并重算指标")

    config = RAGPipelineConfig(refusal_threshold=refusal_threshold)
    pipeline = RAGPipeline(config)

    rows: list[dict] = []
    skipped = 0
    started = time.perf_counter()

    try:
        for index, question in enumerate(questions, start=1):
            if resume and question.id in existing_by_id:
                row = dict(existing_by_id[question.id])
                for key in ("citations_json", "evidence_check_json", "ragas_contexts"):
                    if key in row and isinstance(row[key], str):
                        try:
                            row[key] = json.loads(row[key])
                        except json.JSONDecodeError:
                            pass
                rows.append(row)
                skipped += 1
                continue

            rag_query = RAGQuery(
                query=question.query,
                stock_code=question.stock_code,
                query_type=question.query_type,
            )
            t0 = time.perf_counter()
            result = pipeline.run(rag_query, config=config)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            row = row_from_pipeline(question, result)
            row["latency_ms"] = elapsed_ms
            rows.append(row)

            status = "拒答" if result.refused else "回答"
            hint = _format_refusal_hint(row)
            print(
                f"[{index}/{len(questions)}] {question.id} {status} "
                f"rerank={result.top_rerank_score:.3f} "
                f"citation_acc={row.get('citation_accuracy', 0):.2f} "
                f"refusal_ok={row.get('refusal_correct')}{hint} "
                f"({elapsed_ms}ms)",
                flush=True,
            )
    finally:
        pipeline.close()

    total_s = time.perf_counter() - started
    print(
        f"\nPipeline 评测完成：新跑 {len(rows) - skipped} 题，跳过 {skipped} 题，"
        f"合计 {len(rows)} 题，耗时 {total_s:.1f}s"
    )

    if not skip_ragas:
        print("\n[RAGAS] 运行 faithfulness + answer_relevancy ...")
        try:
            ragas_scores, _ = run_ragas_evaluate(
                rows,
                llm_model=ragas_llm,
                ragas_backend=ragas_backend,
                raise_on_error=False,
            )
            print(
                f"  faithfulness={ragas_scores.get('faithfulness', float('nan')):.4f} "
                f"answer_relevancy={ragas_scores.get('answer_relevancy', float('nan')):.4f} "
                f"(scored_n={ragas_scores.get('ragas_scored_n', 'n/a')})"
            )
        except Exception as exc:
            print(f"[RAGAS 失败] {exc}")
            for row in rows:
                row.setdefault("ragas_faithfulness", None)
                row.setdefault("ragas_answer_relevancy", None)
    else:
        for row in rows:
            row.setdefault("ragas_faithfulness", None)
            row.setdefault("ragas_answer_relevancy", None)

    return rows


def save_reports(
    rows: list[dict],
    *,
    save_detail: bool = False,
) -> None:
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

    if save_detail:
        with open(DETAIL_JSONL, "w", encoding="utf-8") as output_file:
            for row in rows:
                detail = dict(row)
                output_file.write(json.dumps(detail, ensure_ascii=False) + "\n")

    print(f"\n已写入：\n  {RESULTS_CSV}\n  {METRICS_CSV}\n  {METRICS_BY_TYPE_CSV}")
    if save_detail:
        print(f"  {DETAIL_JSONL}")

    print("\n=== 整体指标 ===")
    primary_note = (
        "（refusal_accuracy = evidence 门控口径，与产品一致；"
        "refusal_accuracy_retrieval = 检索 gold 命中口径）"
    )
    print(f"  {primary_note}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}" if value == value else f"  {key}: nan")
        else:
            print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 生成质量评测（150 题）")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--limit", type=int, default=None, help="仅跑前 N 题（调试）")
    parser.add_argument(
        "--refusal-threshold",
        type=float,
        default=DEFAULT_RERANK_REFUSAL_THRESHOLD,
    )
    parser.add_argument("--skip-ragas", action="store_true", help="跳过 RAGAS（仅规则指标）")
    parser.add_argument("--ragas-llm", type=str, default=None, help="RAGAS 评判 LLM，如 qwen3:8b")
    parser.add_argument(
        "--ragas-backend",
        type=str,
        default=None,
        help="RAGAS 后端：ollama|openai|auto（默认 auto→本地 Ollama）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="跳过 eval_generation_results.csv 中已有 question_id",
    )
    parser.add_argument("--save-detail", action="store_true", help="另存 eval_generation_detail.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ensure_stdout_unbuffered()

    questions_path = args.questions.resolve()
    questions = load_questions(questions_path)
    validate_questions(questions, load_chunk_id_set())

    if args.dry_run:
        print(f"[dry-run] 评测题 {len(questions)} 条：{questions_path}")
        print("  python src/eval_generation.py --limit 5 --skip-ragas --save-detail")
        print("  python src/eval_ragas.py   # 本地 RAGAS（需 ollama pull qwen3:8b）")
        return

    rows = run_generation_eval(
        questions,
        refusal_threshold=args.refusal_threshold,
        limit=args.limit,
        skip_ragas=args.skip_ragas,
        ragas_llm=args.ragas_llm,
        ragas_backend=args.ragas_backend,
        save_detail=args.save_detail,
        resume=args.resume,
    )
    save_reports(rows, save_detail=args.save_detail)


if __name__ == "__main__":
    main()
