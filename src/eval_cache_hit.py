#!/usr/bin/env python3
"""
缓存命中效果评测：warmup（原问）→ measure（paraphrase）。

用法:
    PYTHONPATH=src python src/eval_cache_hit.py --modes l1_only,l1_l2 --stock-code-mode empty
    PYTHONPATH=src python src/eval_cache_hit.py --limit 20 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from hf_env import bootstrap_hf_cache

bootstrap_hf_cache()

from eval_cache_common import CacheEvalMode, apply_cache_mode, prepare_isolated_cache_dir, reset_cache_runtime
from eval_generation import _check_indexes, _ensure_generation_ollama, _probe_milvus_available
from rag_pipeline import RAGPipeline, RAGPipelineConfig
from rag_types import RAGQuery

PROJECT_ROOT = CURRENT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "eval"
LOG_DIR = OUTPUT_DIR / "logs"
PAIRS_PATH = OUTPUT_DIR / "cache_hit_pairs.jsonl"


def _setup_file_logger(run_id: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"cache_hit_{run_id}.log"
    logger = logging.getLogger(f"cache_hit_eval.{run_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return logger


def load_pairs(path: Path, *, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit is not None:
        rows = rows[:limit]
    return rows


def _resolve_stock_code(pair: dict, mode: str) -> str:
    if mode == "empty":
        return ""
    if mode == "gold":
        return str(pair.get("gold_stock_code") or pair.get("stock_code") or "")
    return str(pair.get("stock_code") or "")


def _answer_signature(result) -> str:
    refused = getattr(result, "refused", False)
    answer = getattr(result, "answer", "") or ""
    return f"refused={refused}|{answer[:200]}"


def run_pair_phase(
    pipeline: RAGPipeline,
    config: RAGPipelineConfig,
    pair: dict,
    *,
    phase: str,
    query_text: str,
    eval_mode: str,
    stock_code: str,
    logger: logging.Logger,
) -> dict:
    rag_query = RAGQuery(
        query=query_text,
        stock_code=stock_code,
        query_type=pair.get("query_type", "factual"),
    )
    t0 = time.perf_counter()
    result = pipeline.run(rag_query, config=config, use_cache=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    cache = result.cache
    row = {
        "pair_id": pair["pair_id"],
        "seed_id": pair.get("seed_id", ""),
        "variant_type": pair.get("variant_type", ""),
        "phase": phase,
        "mode": eval_mode,
        "query": query_text,
        "expected_layer": pair.get("expected_layer", ""),
        "stock_code": stock_code,
        "latency_ms": round(elapsed_ms, 2),
        "cache_hit": cache.hit if cache else False,
        "cache_source": cache.source if cache else "none",
        "cache_reason": cache.reason if cache else "",
        "cache_similarity": cache.similarity if cache else None,
        "safety_ok": cache.safety_ok if cache else True,
        "safety_reason": cache.safety_reason if cache else "",
        "refused": result.refused,
        "answer_preview": (result.answer or "")[:120],
        "answer_signature": _answer_signature(result),
        "top_rerank_score": result.top_rerank_score,
    }
    logger.info(
        "phase=%s mode=%s pair=%s variant=%s hit=%s source=%s sim=%s refused=%s query=%r",
        phase,
        eval_mode,
        pair["pair_id"],
        pair.get("variant_type"),
        row["cache_hit"],
        row["cache_source"],
        row["cache_similarity"],
        row["refused"],
        query_text[:80],
    )
    return row


def run_mode(
    pairs: list[dict],
    mode: CacheEvalMode,
    *,
    tmp_root: Path,
    stock_code_mode: str,
    logger: logging.Logger,
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
        logger.info("=== mode=%s warmup (%d pairs) ===", mode.name, len(pairs))
        warmup_sigs: dict[str, str] = {}
        for pair in pairs:
            stock = _resolve_stock_code(pair, stock_code_mode)
            row = run_pair_phase(
                pipeline,
                config,
                pair,
                phase="warmup",
                query_text=pair["original_query"],
                eval_mode=mode.name,
                stock_code=stock,
                logger=logger,
            )
            warmup_sigs[pair["pair_id"]] = row["answer_signature"]
            row["warmup_stored"] = row["cache_source"] == "pipeline" or row["phase"] == "warmup"
            all_rows.append(row)

        logger.info("=== mode=%s measure (%d pairs) ===", mode.name, len(pairs))
        for pair in pairs:
            stock = _resolve_stock_code(pair, stock_code_mode)
            row = run_pair_phase(
                pipeline,
                config,
                pair,
                phase="measure",
                query_text=pair["paraphrase_query"],
                eval_mode=mode.name,
                stock_code=stock,
                logger=logger,
            )
            warmup_sig = warmup_sigs.get(pair["pair_id"], "")
            row["answer_match"] = row["answer_signature"] == warmup_sig
            row["expected_layer"] = pair.get("expected_layer", "")
            layer_ok = (
                (pair.get("expected_layer") == "l1_exact" and row["cache_source"] == "l1_exact")
                or (pair.get("expected_layer") == "l2_semantic" and row["cache_source"] == "l2_semantic")
                or row["cache_hit"]
            )
            row["layer_match"] = layer_ok
            all_rows.append(row)
    finally:
        pipeline.close()
        reset_cache_runtime()

    return all_rows


def aggregate_by_variant(measure_rows: list[dict]) -> pd.DataFrame:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in measure_rows:
        groups[row.get("variant_type") or "unknown"].append(row)

    records = []
    for variant, rows in sorted(groups.items()):
        n = len(rows)
        hits = sum(1 for r in rows if r.get("cache_hit"))
        l1 = sum(1 for r in rows if r.get("cache_source") == "l1_exact")
        l2 = sum(1 for r in rows if r.get("cache_source") == "l2_semantic")
        ans_match = sum(1 for r in rows if r.get("answer_match"))
        records.append(
            {
                "variant_type": variant,
                "count": n,
                "hit_rate": round(hits / n, 4) if n else 0.0,
                "l1_hits": l1,
                "l2_hits": l2,
                "answer_match_rate": round(ans_match / n, 4) if n else 0.0,
                "avg_latency_ms": round(
                    sum(r["latency_ms"] for r in rows) / n if n else 0.0,
                    2,
                ),
            }
        )
    return pd.DataFrame(records)


def write_report(
    run_id: str,
    all_rows: list[dict],
    *,
    modes: list[str],
) -> Path:
    measure_rows = [r for r in all_rows if r.get("phase") == "measure"]
    badcases = [
        r
        for r in measure_rows
        if not r.get("cache_hit") or not r.get("answer_match")
    ]

    lines = [
        f"# 缓存命中评测报告 ({run_id})",
        "",
        f"生成时间: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 按模式汇总",
        "",
    ]

    for mode_name in modes:
        mode_measure = [r for r in measure_rows if r.get("mode") == mode_name]
        if not mode_measure:
            continue
        n = len(mode_measure)
        hits = sum(1 for r in mode_measure if r.get("cache_hit"))
        lines.append(f"### {mode_name}")
        lines.append(f"- 命中率: {hits}/{n} ({hits/n:.1%})")
        lines.append(f"- L1: {sum(1 for r in mode_measure if r.get('cache_source')=='l1_exact')}")
        lines.append(f"- L2: {sum(1 for r in mode_measure if r.get('cache_source')=='l2_semantic')}")
        lines.append(f"- 答案一致: {sum(1 for r in mode_measure if r.get('answer_match'))}/{n}")
        lines.append("")

        df = aggregate_by_variant(mode_measure)
        if not df.empty:
            headers = list(df.columns)
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for _, row in df.iterrows():
                lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
            lines.append("")

    lines.append("## Badcases")
    lines.append("")
    if not badcases:
        lines.append("无 badcase。")
    else:
        lines.append("| pair_id | mode | variant | hit | source | sim | answer_match | query |")
        lines.append("|---------|------|---------|-----|--------|-----|--------------|-------|")
        for r in badcases[:50]:
            lines.append(
                f"| {r['pair_id']} | {r['mode']} | {r.get('variant_type')} | "
                f"{r.get('cache_hit')} | {r.get('cache_source')} | {r.get('cache_similarity')} | "
                f"{r.get('answer_match')} | {r.get('query','')[:40]} |"
            )
        if len(badcases) > 50:
            lines.append(f"\n... 另有 {len(badcases)-50} 条 badcase，见 CSV。")

    report_path = OUTPUT_DIR / f"cache_hit_report_{run_id}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_modes(raw: str) -> list[CacheEvalMode]:
    mapping = {
        "l1_only": CacheEvalMode.l1_only(),
        "l1_l2": CacheEvalMode.l1_l2(),
    }
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return [mapping[k] for k in keys]


def main() -> int:
    parser = argparse.ArgumentParser(description="缓存命中 paraphrase 评测")
    parser.add_argument("--pairs", type=Path, default=PAIRS_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--modes", default="l1_only,l1_l2")
    parser.add_argument("--stock-code-mode", choices=["empty", "gold", "pair"], default="empty")
    parser.add_argument("--refusal-threshold", type=float, default=0.35)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--generate-pairs", action="store_true", help="运行前重新生成 pairs 文件")
    args = parser.parse_args()

    if args.generate_pairs or not args.pairs.exists():
        from scripts.generate_cache_hit_pairs import main as gen_main

        gen_main()

    pairs = load_pairs(args.pairs, limit=args.limit)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger = _setup_file_logger(run_id)

    logger.info("cache_hit_eval start run_id=%s pairs=%d modes=%s", run_id, len(pairs), args.modes)

    if args.dry_run:
        logger.info("dry-run: pairs=%d first=%s", len(pairs), pairs[0] if pairs else None)
        return 0

    _check_indexes()
    try:
        if not _probe_milvus_available():
            logger.warning("主 Milvus 探针失败（可能被其他进程占用），继续尝试 Pipeline")
    except Exception as exc:
        logger.warning("主 Milvus 探针异常: %s，继续尝试 Pipeline", exc)
    _ensure_generation_ollama()

    modes = parse_modes(args.modes)
    tmp_root = OUTPUT_DIR / "tmp_cache_hit" / run_id
    tmp_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for mode in modes:
        rows = run_mode(
            pairs,
            mode,
            tmp_root=tmp_root,
            stock_code_mode=args.stock_code_mode,
            logger=logger,
            refusal_threshold=args.refusal_threshold,
        )
        all_rows.extend(rows)

    csv_path = OUTPUT_DIR / f"cache_hit_results_{run_id}.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    report_path = write_report(run_id, all_rows, modes=[m.name for m in modes])

    measure = [r for r in all_rows if r.get("phase") == "measure"]
    total_hits = sum(1 for r in measure if r.get("cache_hit"))
    logger.info(
        "cache_hit_eval done run_id=%s measure=%d hits=%d rate=%.1f%% csv=%s report=%s",
        run_id,
        len(measure),
        total_hits,
        100.0 * total_hits / len(measure) if measure else 0.0,
        csv_path,
        report_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
