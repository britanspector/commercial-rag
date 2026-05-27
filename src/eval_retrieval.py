"""
检索离线评测：向量 / BM25 / 混合三路召回，计算 Recall@K / MRR。

前置条件：
    1. python src/chunk_mineru.py
    2. python src/embed_chunks.py
    3. python src/build_bm25_index.py

用法：
    python src/eval_retrieval.py --dry-run
    python src/eval_retrieval.py --route vector
    python src/eval_retrieval.py --compare-routes
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

PROJECT_ROOT = CURRENT_DIR.parent
DEFAULT_QUESTIONS = PROJECT_ROOT / "data" / "eval" / "eval_questions.jsonl"
CHUNKS_JSONL = PROJECT_ROOT / "data" / "parsed" / "chunks.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "eval"

ROUTE_LABELS = {
    "vector": "路线A-纯向量",
    "bm25": "路线B-纯BM25",
    "hybrid": "路线C-混合",
}


@dataclass
class EvalQuestion:
    id: str
    query: str
    gold_answer: str = ""
    category: str = ""
    query_type: str = "factual"
    stock_code: str = ""
    doc_id: str = ""
    industry_label: str = ""
    section_keywords: list[str] = field(default_factory=list)
    must_contain_any: list[str] = field(default_factory=list)
    gold_chunk_ids: list[str] = field(default_factory=list)
    negative_stock_codes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> EvalQuestion:
        return cls(
            id=str(data["id"]),
            query=str(data["query"]),
            gold_answer=str(data.get("gold_answer", "")),
            category=str(data.get("category", "")),
            query_type=str(data.get("query_type", "factual")),
            stock_code=str(data.get("stock_code", "")).strip(),
            doc_id=str(data.get("doc_id", "")).strip(),
            industry_label=str(data.get("industry_label", "")).strip(),
            section_keywords=list(data.get("section_keywords") or []),
            must_contain_any=list(data.get("must_contain_any") or []),
            gold_chunk_ids=list(data.get("gold_chunk_ids") or []),
            negative_stock_codes=list(data.get("negative_stock_codes") or []),
        )


def load_questions(path: Path) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    with open(path, "r", encoding="utf-8") as input_file:
        for line_no, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                questions.append(EvalQuestion.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"评测题格式错误 {path}:{line_no}: {error}") from error
    if not questions:
        raise ValueError(f"评测集为空：{path}")
    return questions


def load_chunk_id_set() -> set[str]:
    if not CHUNKS_JSONL.exists():
        return set()

    chunk_ids: set[str] = set()
    with open(CHUNKS_JSONL, "r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if line:
                chunk_ids.add(json.loads(line)["chunk_id"])
    return chunk_ids


def encode_query(model, query: str) -> list[float]:
    from embed_chunks import BGE_QUERY_PREFIX, NORMALIZE_EMBEDDINGS

    query_text = query if query.startswith(BGE_QUERY_PREFIX) else f"{BGE_QUERY_PREFIX}{query}"
    vector = model.encode(
        [query_text],
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        convert_to_numpy=True,
    )[0]
    return vector.tolist()


MUST_TOKEN_ALIASES: dict[str, list[str]] = {
    "EPS": ["EPS", "每股收益", "摊薄每股收益"],
    "YoY": ["YoY", "同比"],
    "PE": ["PE", "市盈率"],
}


def hit_blob(hit: dict) -> str:
    parts = [
        hit.get("section_title", ""),
        hit.get("text", ""),
        hit.get("display_name", ""),
        hit.get("company_name", ""),
        hit.get("report_title", ""),
        hit.get("broker", ""),
        hit.get("rating", ""),
    ]
    return "\n".join(part for part in parts if part)


def expand_must_tokens(tokens: list[str]) -> list[str]:
    expanded: list[str] = []
    for token in tokens:
        expanded.extend(MUST_TOKEN_ALIASES.get(token, [token]))
    return expanded


def section_keyword_matches(hit: dict, keywords: list[str]) -> bool:
    blob = hit_blob(hit)
    rating = str(hit.get("rating", "")).strip()
    for keyword in keywords:
        if keyword in blob:
            return True
        if keyword in ("评级", "投资评级") and rating:
            return True
        if keyword in ("买入", "增持") and (
            keyword in blob or keyword in rating or f"{keyword}-" in blob
        ):
            return True
    return False


def is_hit_relevant(hit: dict, question: EvalQuestion) -> bool:
    chunk_id = hit.get("chunk_id", "")
    stock_code = str(hit.get("stock_code", "")).strip()
    doc_id = str(hit.get("doc_id", "")).strip()
    industry_label = str(hit.get("industry_label", "")).strip()

    if question.negative_stock_codes and stock_code in question.negative_stock_codes:
        return False

    if question.gold_chunk_ids and chunk_id in question.gold_chunk_ids:
        return True

    # 汇总型：无单一股票时，按行业标签 + 关键词约束
    if question.query_type == "summary" and not question.stock_code and not question.doc_id:
        if question.industry_label and industry_label != question.industry_label:
            return False
        if question.section_keywords and not section_keyword_matches(
            hit, question.section_keywords
        ):
            return False
        if question.must_contain_any:
            blob = hit_blob(hit)
            candidates = expand_must_tokens(question.must_contain_any)
            if not any(token in blob for token in candidates):
                return False
        return True

    stock_ok = not question.stock_code or stock_code == question.stock_code
    doc_ok = not question.doc_id or doc_id == question.doc_id
    if not stock_ok and not doc_ok:
        return False

    if question.section_keywords and not section_keyword_matches(
        hit, question.section_keywords
    ):
        return False

    if question.must_contain_any:
        blob = hit_blob(hit)
        candidates = expand_must_tokens(question.must_contain_any)
        if not any(token in blob for token in candidates):
            return False

    return True


def recall_at_k(relevant_ranks: list[int], k: int) -> float:
    return 1.0 if any(rank <= k for rank in relevant_ranks) else 0.0


def mrr(relevant_ranks: list[int]) -> float:
    if not relevant_ranks:
        return 0.0
    return 1.0 / min(relevant_ranks)


def evaluate_hits(
    question: EvalQuestion,
    hits: list[dict],
    route: str,
) -> dict:
    relevant_ranks = [
        rank
        for rank, hit in enumerate(hits, start=1)
        if is_hit_relevant(hit, question)
    ]
    first_rank = min(relevant_ranks) if relevant_ranks else None

    return {
        "question_id": question.id,
        "query_type": question.query_type,
        "category": question.category,
        "route": route,
        "query": question.query,
        "gold_answer": question.gold_answer,
        "stock_code": question.stock_code,
        "doc_id": question.doc_id,
        "hit": first_rank is not None,
        "first_relevant_rank": first_rank or "",
        "recall_at_3": recall_at_k(relevant_ranks, 3),
        "recall_at_5": recall_at_k(relevant_ranks, 5),
        "recall_at_10": recall_at_k(relevant_ranks, 10),
        "mrr": mrr(relevant_ranks),
        "top1_chunk_id": hits[0].get("chunk_id", "") if hits else "",
        "top1_stock_code": hits[0].get("stock_code", "") if hits else "",
        "top1_section": hits[0].get("section_title", "") if hits else "",
        "top1_display_name": hits[0].get("display_name", "") if hits else "",
        "relevant_chunks": "|".join(
            hits[rank - 1]["chunk_id"] for rank in relevant_ranks[:5]
        ),
    }


def aggregate_metrics(results: list[dict]) -> dict[str, float]:
    n = len(results)
    return {
        "question_count": float(n),
        "recall_at_3": sum(row["recall_at_3"] for row in results) / n,
        "recall_at_5": sum(row["recall_at_5"] for row in results) / n,
        "recall_at_10": sum(row["recall_at_10"] for row in results) / n,
        "mrr": sum(row["mrr"] for row in results) / n,
        "hit_rate": sum(1 for row in results if row["hit"]) / n,
    }


def validate_questions(questions: list[EvalQuestion], chunk_ids: set[str]) -> None:
    missing_gold: list[str] = []
    for question in questions:
        for chunk_id in question.gold_chunk_ids:
            if chunk_id not in chunk_ids:
                missing_gold.append(f"{question.id}:{chunk_id}")

    type_counts: dict[str, int] = {}
    for question in questions:
        type_counts[question.query_type] = type_counts.get(question.query_type, 0) + 1

    print(f"评测题数量：{len(questions)}")
    print(f"query_type 分布：{type_counts}")
    print(f"chunks.jsonl chunk 数：{len(chunk_ids)}")
    if missing_gold:
        print(f"[警告] {len(missing_gold)} 个 gold_chunk_id 在 chunks.jsonl 中不存在：")
        for item in missing_gold[:10]:
            print(f"  - {item}")
    else:
        print("gold_chunk_id 校验：全部存在")


def run_retrieval_eval(
    questions: list[EvalQuestion],
    route: str,
    top_k: int,
    hybrid_vector_weight: float,
) -> tuple[list[dict], dict[str, float]]:
    from embed_chunks import (
        EMBED_DIM,
        EMBED_MODEL,
        OUTPUT_MILVUS_DB,
        load_embedder,
        resolve_device,
    )
    from retrieval import HybridRetriever, RecallRoute

    if not OUTPUT_MILVUS_DB.exists():
        raise FileNotFoundError(
            f"未找到 Milvus 数据库，请先运行 src/embed_chunks.py\n{OUTPUT_MILVUS_DB}"
        )

    device = resolve_device()
    print(f"Embedding 模型：{EMBED_MODEL}")
    print(f"设备：{device}")
    print(f"召回路线：{ROUTE_LABELS.get(route, route)}")
    print(f"Top-K：{top_k}")
    if route == "hybrid":
        print(f"混合权重（向量）：{hybrid_vector_weight}")

    model = load_embedder(device)
    retriever = HybridRetriever.from_paths(
        OUTPUT_MILVUS_DB,
        vector_dim=EMBED_DIM,
        hybrid_vector_weight=hybrid_vector_weight,
    )
    if not retriever.milvus_store.has_collection():
        retriever.close()
        raise FileNotFoundError("Milvus collection 不存在，请先运行 src/embed_chunks.py")

    row_count = retriever.milvus_store.count()
    retriever.milvus_store.load()
    print(f"Milvus 向量数：{row_count}")
    print(f"BM25 文档数：{len(retriever.bm25_index.chunk_ids)}")

    recall_route = RecallRoute(route)
    results: list[dict] = []
    for question in questions:
        query_vector = encode_query(model, question.query)
        hits = retriever.retrieve(recall_route, question.query, query_vector, top_k)
        results.append(evaluate_hits(question, hits, route))

    retriever.close()
    del model

    metrics = aggregate_metrics(results)
    return results, metrics


def save_reports(
    results: list[dict],
    metrics: dict[str, float],
    route: str,
    top_k: int,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = OUTPUT_DIR / f"eval_results_{route}.csv"
    metrics_path = OUTPUT_DIR / f"eval_metrics_{route}.csv"
    misses_path = OUTPUT_DIR / f"eval_misses_{route}.jsonl"

    pd.DataFrame(results).to_csv(results_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([{**metrics, "route": route, "top_k": top_k}]).to_csv(
        metrics_path, index=False, encoding="utf-8-sig"
    )

    misses = [row for row in results if not row["hit"]]
    with open(misses_path, "w", encoding="utf-8") as output_file:
        for row in misses:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print(f"检索评测结果 — {ROUTE_LABELS.get(route, route)}")
    print(f"Recall@3：  {metrics['recall_at_3']:.1%}")
    print(f"Recall@5：  {metrics['recall_at_5']:.1%}")
    print(f"Recall@10： {metrics['recall_at_10']:.1%}")
    print(f"MRR：       {metrics['mrr']:.3f}")
    print(
        f"命中率：    {metrics['hit_rate']:.1%} "
        f"({int(metrics['hit_rate'] * metrics['question_count'])}/"
        f"{int(metrics['question_count'])})"
    )
    print(f"明细 CSV：  {results_path}")
    print(f"指标 CSV：  {metrics_path}")
    print(f"未命中：    {misses_path} ({len(misses)} 题)")
    print("=" * 70)


def save_route_comparison(
    comparison_rows: list[dict],
    per_type_rows: list[dict] | None = None,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = OUTPUT_DIR / "eval_route_comparison.csv"
    pd.DataFrame(comparison_rows).to_csv(
        comparison_path, index=False, encoding="utf-8-sig"
    )
    print(f"\n三路对比表：{comparison_path}")
    print(pd.DataFrame(comparison_rows).to_string(index=False))

    if per_type_rows:
        type_path = OUTPUT_DIR / "eval_route_comparison_by_query_type.csv"
        pd.DataFrame(per_type_rows).to_csv(type_path, index=False, encoding="utf-8-sig")
        print(f"分 query_type 对比：{type_path}")


def run_compare_routes(
    questions: list[EvalQuestion],
    top_k: int,
    hybrid_vector_weight: float,
) -> None:
    from embed_chunks import (
        EMBED_DIM,
        EMBED_MODEL,
        OUTPUT_MILVUS_DB,
        load_embedder,
        resolve_device,
    )
    from retrieval import HybridRetriever, RecallRoute

    device = resolve_device()
    print(f"Embedding 模型：{EMBED_MODEL}")
    print(f"设备：{device}")
    print(f"Top-K：{top_k}")

    model = load_embedder(device)
    retriever = HybridRetriever.from_paths(
        OUTPUT_MILVUS_DB,
        vector_dim=EMBED_DIM,
        hybrid_vector_weight=hybrid_vector_weight,
    )
    retriever.milvus_store.load()
    print(f"Milvus 向量数：{retriever.milvus_store.count()}")
    print(f"BM25 文档数：{len(retriever.bm25_index.chunk_ids)}")

    comparison_rows: list[dict] = []
    per_type_rows: list[dict] = []

    for route in ("vector", "bm25", "hybrid"):
        print(f"\n>>> {ROUTE_LABELS[route]}")
        recall_route = RecallRoute(route)
        results: list[dict] = []
        for question in questions:
            query_vector = encode_query(model, question.query)
            hits = retriever.retrieve(
                recall_route, question.query, query_vector, top_k
            )
            results.append(evaluate_hits(question, hits, route))
        metrics = aggregate_metrics(results)
        save_reports(results, metrics, route, top_k)
        comparison_rows.append(
            {
                "route": route,
                "route_label": ROUTE_LABELS[route],
                "question_count": int(metrics["question_count"]),
                "recall_at_3": round(metrics["recall_at_3"], 4),
                "recall_at_5": round(metrics["recall_at_5"], 4),
                "recall_at_10": round(metrics["recall_at_10"], 4),
                "mrr": round(metrics["mrr"], 4),
                "hit_rate": round(metrics["hit_rate"], 4),
                "top_k": top_k,
                "hybrid_vector_weight": hybrid_vector_weight if route == "hybrid" else "",
            }
        )

        for query_type in ("factual", "comparative", "summary"):
            subset = [row for row in results if row["query_type"] == query_type]
            if not subset:
                continue
            subset_metrics = aggregate_metrics(subset)
            per_type_rows.append(
                {
                    "route": route,
                    "query_type": query_type,
                    "question_count": len(subset),
                    "recall_at_3": round(subset_metrics["recall_at_3"], 4),
                    "recall_at_5": round(subset_metrics["recall_at_5"], 4),
                    "recall_at_10": round(subset_metrics["recall_at_10"], 4),
                    "mrr": round(subset_metrics["mrr"], 4),
                }
            )

    retriever.close()
    del model
    save_route_comparison(comparison_rows, per_type_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索离线评测（向量/BM25/混合）")
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help="评测题 JSONL 路径",
    )
    parser.add_argument("--top-k", type=int, default=10, help="检索 Top-K")
    parser.add_argument(
        "--route",
        choices=["vector", "bm25", "hybrid"],
        default="vector",
        help="单一路线评测",
    )
    parser.add_argument(
        "--compare-routes",
        action="store_true",
        help="依次评测 vector / bm25 / hybrid 并输出对比表",
    )
    parser.add_argument(
        "--hybrid-vector-weight",
        type=float,
        default=0.5,
        help="混合检索中向量分权重（默认 0.5）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅校验评测题与 chunks.jsonl，不访问 Milvus",
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)
    chunk_ids = load_chunk_id_set()
    validate_questions(questions, chunk_ids)

    if args.dry_run:
        print("\n[dry-run] 跳过检索。完成 embed + BM25 后请运行：")
        print("  python src/build_bm25_index.py")
        print("  python src/eval_retrieval.py --compare-routes")
        return

    if args.compare_routes:
        run_compare_routes(questions, args.top_k, args.hybrid_vector_weight)
        return

    results, metrics = run_retrieval_eval(
        questions, args.route, args.top_k, args.hybrid_vector_weight
    )
    save_reports(results, metrics, args.route, top_k=args.top_k)


if __name__ == "__main__":
    main()
