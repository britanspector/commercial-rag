"""
comparative 专项消融评测。

目标：
1. 在 comparative 100 题集上比较不同 retrieval 变体的 Recall/MRR
2. 同时记录 Top-5 双公司覆盖，解释 Recall 与 coverage 的 trade-off

变体说明：
- vector: 纯向量
- bm25: 纯 BM25
- hybrid_plain: 原问题直接做 hybrid，不走 comparative 子查询 / RRF
- hybrid_rrf_fallback: 使用 retrieval.py 的 comparative fallback（extract entities + 共享原 query_vector）
- hybrid_subq_shared: 使用清洗后的 comparative 子查询，但共享原 query_vector
- hybrid_subq_indep: 使用清洗后的 comparative 子查询，且每个子查询独立 embedding（最接近当前 comparative 生产召回）
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import time

import pandas as pd

CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

PROJECT_ROOT = CURRENT_DIR.parent
DEFAULT_QUESTIONS = PROJECT_ROOT / "data" / "eval" / "eval_questions_comparative_100.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "eval" / "comparative_100"

from bm25_store import DEFAULT_INDEX_PATH
from embed_chunks import EMBED_DIM, EMBED_MODEL, OUTPUT_MILVUS_DB, load_embedder, resolve_device
from eval_retrieval import (
    EvalQuestion,
    aggregate_metrics,
    encode_query,
    evaluate_hits,
    load_chunk_id_set,
    validate_questions,
)
from query_enhance import build_comparative_sub_queries, extract_compare_entities
from rag_types import EntitySubQuery
from retrieval import DEFAULT_HYBRID_POOL_SIZE, DEFAULT_HYBRID_VECTOR_WEIGHT, HybridRetriever, RecallRoute


VARIANT_LABELS = {
    "vector": "纯向量",
    "bm25": "纯BM25",
    "hybrid_plain": "原问题直接Hybrid",
    "hybrid_rrf_fallback": "旧式RRF（共享原query向量）",
    "hybrid_subq_shared": "清洗子查询+共享原query向量",
    "hybrid_subq_indep": "清洗子查询+独立向量+RRF",
}


@dataclass
class ComparativeQuestion:
    eval_question: EvalQuestion
    compare_tag: str = "legacy_existing"
    source_question_ids: list[str] | None = None


def load_questions(path: Path) -> list[ComparativeQuestion]:
    rows: list[ComparativeQuestion] = []
    with open(path, "r", encoding="utf-8") as input_file:
        for line_no, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"问题文件格式错误 {path}:{line_no}: {error}") from error
            question = EvalQuestion.from_dict(data)
            rows.append(
                ComparativeQuestion(
                    eval_question=question,
                    compare_tag=str(data.get("compare_tag", "legacy_existing")),
                    source_question_ids=list(data.get("source_question_ids") or []),
                )
            )
    if not rows:
        raise ValueError(f"问题文件为空：{path}")
    return rows


def distinct_companies(hits: list[dict], top_k: int = 5) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for hit in hits[:top_k]:
        company = str(hit.get("company_name", "")).strip()
        if company and company not in seen:
            names.append(company)
            seen.add(company)
    return names


def build_sub_queries(
    query: str,
    query_vector: list[float],
    *,
    embedder,
    use_clean_sub_queries: bool,
    independent_vectors: bool,
) -> list[EntitySubQuery]:
    entities = extract_compare_entities(query)
    if len(entities) < 2:
        return []

    if use_clean_sub_queries:
        pairs = build_comparative_sub_queries(query, entities)
    else:
        pairs = [(entity, f"{entity} {query}") for entity in entities[:3]]

    sub_queries: list[EntitySubQuery] = []
    for entity, sub_query in pairs:
        sub_vector = encode_query(embedder, sub_query) if independent_vectors else None
        sub_queries.append(EntitySubQuery(entity=entity, query=sub_query, query_vector=sub_vector))
    return sub_queries


def retrieve_for_variant(
    retriever: HybridRetriever,
    *,
    variant: str,
    query: str,
    query_vector: list[float],
    top_k: int,
    stock_code: str,
    embedder,
) -> list[dict]:
    if variant == "vector":
        return retriever.retrieve(
            RecallRoute.VECTOR,
            query,
            query_vector,
            top_k,
            stock_code=stock_code,
            query_type="comparative",
        )
    if variant == "bm25":
        return retriever.retrieve(
            RecallRoute.BM25,
            query,
            query_vector,
            top_k,
            stock_code=stock_code,
            query_type="comparative",
        )
    if variant == "hybrid_plain":
        return retriever._retrieve_hybrid_once(
            query,
            query_vector,
            top_k,
            stock_code=stock_code,
            query_type="comparative",
        )
    if variant == "hybrid_rrf_fallback":
        return retriever.retrieve(
            RecallRoute.HYBRID,
            query,
            query_vector,
            top_k,
            stock_code=stock_code,
            query_type="comparative",
            entity_sub_queries=None,
        )
    if variant == "hybrid_subq_shared":
        sub_queries = build_sub_queries(
            query,
            query_vector,
            embedder=embedder,
            use_clean_sub_queries=True,
            independent_vectors=False,
        )
        return retriever.retrieve(
            RecallRoute.HYBRID,
            query,
            query_vector,
            top_k,
            stock_code=stock_code,
            query_type="comparative",
            entity_sub_queries=sub_queries or None,
        )
    if variant == "hybrid_subq_indep":
        sub_queries = build_sub_queries(
            query,
            query_vector,
            embedder=embedder,
            use_clean_sub_queries=True,
            independent_vectors=True,
        )
        return retriever.retrieve(
            RecallRoute.HYBRID,
            query,
            query_vector,
            top_k,
            stock_code=stock_code,
            query_type="comparative",
            entity_sub_queries=sub_queries or None,
        )
    raise ValueError(f"未知 variant: {variant}")


def run_variant_eval(
    questions: list[ComparativeQuestion],
    *,
    variant: str,
    top_k: int,
    hybrid_vector_weight: float,
    hybrid_pool_size: int,
) -> tuple[list[dict], dict[str, float], list[dict]]:
    device = resolve_device()
    print(f"[{variant}] Embedding 模型：{EMBED_MODEL}")
    print(f"[{variant}] 设备：{device}")
    print(f"[{variant}] Top-K：{top_k}")
    print(f"[{variant}] 混合向量权重：{hybrid_vector_weight}")

    embedder = load_embedder(device)
    last_error: Exception | None = None
    retriever: HybridRetriever | None = None
    for attempt in range(5):
        try:
            retriever = HybridRetriever.from_paths(
                OUTPUT_MILVUS_DB,
                vector_dim=EMBED_DIM,
                hybrid_vector_weight=hybrid_vector_weight,
                hybrid_pool_size=hybrid_pool_size,
            )
            break
        except Exception as error:  # pragma: no cover - retry wrapper
            last_error = error
            wait_seconds = 3 * (attempt + 1)
            print(f"[{variant}] Milvus 打开失败，{wait_seconds}s 后重试：{error}")
            time.sleep(wait_seconds)
    if retriever is None:
        assert last_error is not None
        raise last_error
    retriever.milvus_store.load()

    detail_rows: list[dict] = []
    for item in questions:
        question = item.eval_question
        query_vector = encode_query(embedder, question.query)
        hits = retrieve_for_variant(
            retriever,
            variant=variant,
            query=question.query,
            query_vector=query_vector,
            top_k=top_k,
            stock_code=question.stock_code,
            embedder=embedder,
        )
        row = evaluate_hits(question, hits, variant)
        companies = distinct_companies(hits, top_k=5)
        row["variant"] = variant
        row["variant_label"] = VARIANT_LABELS[variant]
        row["compare_tag"] = item.compare_tag
        row["source_question_ids"] = ",".join(item.source_question_ids or [])
        row["distinct_companies_top5"] = len(companies)
        row["top5_companies"] = "|".join(companies)
        row["comparative_insufficient"] = len(companies) < 2
        detail_rows.append(row)

    retriever.close()
    del embedder

    metrics = aggregate_metrics(detail_rows)
    metrics["variant"] = variant
    metrics["variant_label"] = VARIANT_LABELS[variant]
    metrics["comparative_insufficient_rate"] = sum(
        1 for row in detail_rows if row["comparative_insufficient"]
    ) / len(detail_rows)
    metrics["multi_company_top5_rate"] = sum(
        1 for row in detail_rows if row["distinct_companies_top5"] >= 2
    ) / len(detail_rows)
    metrics["avg_distinct_companies_top5"] = sum(
        int(row["distinct_companies_top5"]) for row in detail_rows
    ) / len(detail_rows)

    by_tag_rows: list[dict] = []
    for tag in sorted({row["compare_tag"] for row in detail_rows}):
        subset = [row for row in detail_rows if row["compare_tag"] == tag]
        subset_metrics = aggregate_metrics(subset)
        by_tag_rows.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "compare_tag": tag,
                "question_count": len(subset),
                "recall_at_3": round(subset_metrics["recall_at_3"], 4),
                "recall_at_5": round(subset_metrics["recall_at_5"], 4),
                "recall_at_10": round(subset_metrics["recall_at_10"], 4),
                "mrr": round(subset_metrics["mrr"], 4),
                "hit_rate": round(subset_metrics["hit_rate"], 4),
                "comparative_insufficient_rate": round(
                    sum(1 for row in subset if row["comparative_insufficient"]) / len(subset), 4
                ),
            }
        )
    return detail_rows, metrics, by_tag_rows


def save_outputs(
    *,
    output_dir: Path,
    summary_rows: list[dict],
    by_tag_rows: list[dict],
    detail_rows_by_variant: dict[str, list[dict]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "comparative_ablation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(by_tag_rows).to_csv(
        output_dir / "comparative_ablation_by_tag.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for variant, rows in detail_rows_by_variant.items():
        pd.DataFrame(rows).to_csv(
            output_dir / f"comparative_results_{variant}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        misses = [row for row in rows if not row["hit"]]
        with open(output_dir / f"comparative_misses_{variant}.jsonl", "w", encoding="utf-8") as f:
            for row in misses:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="comparative 专项消融评测")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hybrid-vector-weight", type=float, default=DEFAULT_HYBRID_VECTOR_WEIGHT)
    parser.add_argument("--hybrid-pool-size", type=int, default=DEFAULT_HYBRID_POOL_SIZE)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "vector",
            "bm25",
            "hybrid_plain",
            "hybrid_rrf_fallback",
            "hybrid_subq_shared",
            "hybrid_subq_indep",
        ],
        choices=list(VARIANT_LABELS.keys()),
    )
    args = parser.parse_args()

    if not OUTPUT_MILVUS_DB.exists():
        raise FileNotFoundError(f"未找到 Milvus 数据库：{OUTPUT_MILVUS_DB}")
    if not DEFAULT_INDEX_PATH.exists():
        raise FileNotFoundError(f"未找到 BM25 索引：{DEFAULT_INDEX_PATH}")

    questions = load_questions(args.questions)
    eval_questions = [item.eval_question for item in questions]
    validate_questions(eval_questions, load_chunk_id_set())

    print(f"comparative 题数：{len(questions)}")
    print("行业分布：", dict(sorted(Counter(q.eval_question.industry_label for q in questions).items())))
    print("tag 分布：", dict(sorted(Counter(q.compare_tag for q in questions).items())))

    summary_rows: list[dict] = []
    by_tag_rows: list[dict] = []
    detail_rows_by_variant: dict[str, list[dict]] = {}

    for variant in args.variants:
        rows, metrics, tag_rows = run_variant_eval(
            questions,
            variant=variant,
            top_k=args.top_k,
            hybrid_vector_weight=args.hybrid_vector_weight,
            hybrid_pool_size=args.hybrid_pool_size,
        )
        detail_rows_by_variant[variant] = rows
        summary_rows.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "question_count": int(metrics["question_count"]),
                "recall_at_3": round(metrics["recall_at_3"], 4),
                "recall_at_5": round(metrics["recall_at_5"], 4),
                "recall_at_10": round(metrics["recall_at_10"], 4),
                "mrr": round(metrics["mrr"], 4),
                "hit_rate": round(metrics["hit_rate"], 4),
                "multi_company_top5_rate": round(metrics["multi_company_top5_rate"], 4),
                "comparative_insufficient_rate": round(metrics["comparative_insufficient_rate"], 4),
                "avg_distinct_companies_top5": round(metrics["avg_distinct_companies_top5"], 4),
                "top_k": args.top_k,
                "hybrid_vector_weight": args.hybrid_vector_weight,
                "hybrid_pool_size": args.hybrid_pool_size,
            }
        )
        by_tag_rows.extend(tag_rows)

    save_outputs(
        output_dir=args.output_dir,
        summary_rows=summary_rows,
        by_tag_rows=by_tag_rows,
        detail_rows_by_variant=detail_rows_by_variant,
    )

    print("\ncomparative ablation summary:")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\n输出目录：{args.output_dir}")


if __name__ == "__main__":
    main()
