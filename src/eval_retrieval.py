"""
检索离线评测：对 eval_questions.jsonl 批量跑向量检索，计算 Recall@K / MRR。

前置条件：
    1. python src/chunk_mineru.py
    2. python src/embed_chunks.py

用法：
    python src/eval_retrieval.py
    python src/eval_retrieval.py --dry-run
    python src/eval_retrieval.py --top-k 10
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


@dataclass
class EvalQuestion:
    id: str
    query: str
    gold_answer: str = ""
    category: str = ""
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


# must_contain 字面量扩展为中文研报常用表述
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

    if question.negative_stock_codes and stock_code in question.negative_stock_codes:
        return False

    if question.gold_chunk_ids and chunk_id in question.gold_chunk_ids:
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


def evaluate_hits(question: EvalQuestion, hits: list[dict]) -> dict:
    relevant_ranks = [
        rank
        for rank, hit in enumerate(hits, start=1)
        if is_hit_relevant(hit, question)
    ]
    first_rank = min(relevant_ranks) if relevant_ranks else None

    return {
        "question_id": question.id,
        "category": question.category,
        "query": question.query,
        "gold_answer": question.gold_answer,
        "stock_code": question.stock_code,
        "doc_id": question.doc_id,
        "hit": first_rank is not None,
        "first_relevant_rank": first_rank or "",
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


def validate_questions(questions: list[EvalQuestion], chunk_ids: set[str]) -> None:
    missing_gold: list[str] = []
    for question in questions:
        for chunk_id in question.gold_chunk_ids:
            if chunk_id not in chunk_ids:
                missing_gold.append(f"{question.id}:{chunk_id}")

    print(f"评测题数量：{len(questions)}")
    print(f"chunks.jsonl chunk 数：{len(chunk_ids)}")
    if missing_gold:
        print(f"[警告] {len(missing_gold)} 个 gold_chunk_id 在 chunks.jsonl 中不存在：")
        for item in missing_gold[:10]:
            print(f"  - {item}")
    else:
        print("gold_chunk_id 校验：全部存在")


def run_retrieval_eval(
    questions: list[EvalQuestion],
    top_k: int,
) -> tuple[list[dict], dict[str, float]]:
    from embed_chunks import (
        EMBED_DIM,
        EMBED_MODEL,
        OUTPUT_MILVUS_DB,
        load_embedder,
        resolve_device,
    )
    from milvus_store import MilvusChunkStore

    if not OUTPUT_MILVUS_DB.exists():
        raise FileNotFoundError(
            f"未找到 Milvus 数据库，请先运行 src/embed_chunks.py\n{OUTPUT_MILVUS_DB}"
        )

    device = resolve_device()
    print(f"Embedding 模型：{EMBED_MODEL}")
    print(f"设备：{device}")
    print(f"Top-K：{top_k}")

    model = load_embedder(device)
    store = MilvusChunkStore(OUTPUT_MILVUS_DB, vector_dim=EMBED_DIM)
    if not store.has_collection():
        store.close()
        raise FileNotFoundError("Milvus collection 不存在，请先运行 src/embed_chunks.py")

    row_count = store.count()
    store.load()
    print(f"Milvus 向量数：{row_count}")

    output_fields = [
        "chunk_id",
        "doc_id",
        "filename",
        "display_name",
        "company_name",
        "report_title",
        "broker",
        "industry_label",
        "source_pdf_path",
        "section_title",
        "text",
        "page_start",
        "page_end",
        "contains_table",
        "stock_code",
        "rating",
    ]

    results: list[dict] = []
    for question in questions:
        query_vector = encode_query(model, question.query)
        hits = store.search(query_vector, top_k=top_k, output_fields=output_fields)
        results.append(evaluate_hits(question, hits))

    store.close()
    del model

    n = len(results)
    metrics = {
        "question_count": float(n),
        "recall_at_5": sum(row["recall_at_5"] for row in results) / n,
        "recall_at_10": sum(row["recall_at_10"] for row in results) / n,
        "mrr": sum(row["mrr"] for row in results) / n,
        "hit_rate": sum(1 for row in results if row["hit"]) / n,
    }
    return results, metrics


def save_reports(results: list[dict], metrics: dict[str, float], top_k: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = OUTPUT_DIR / "eval_results.csv"
    metrics_path = OUTPUT_DIR / "eval_metrics.csv"
    misses_path = OUTPUT_DIR / "eval_misses.jsonl"
    detail_path = OUTPUT_DIR / "eval_detail.jsonl"

    pd.DataFrame(results).to_csv(results_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([{**metrics, "top_k": top_k}]).to_csv(
        metrics_path, index=False, encoding="utf-8-sig"
    )

    misses = [row for row in results if not row["hit"]]
    with open(misses_path, "w", encoding="utf-8") as output_file:
        for row in misses:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(detail_path, "w", encoding="utf-8") as output_file:
        for row in results:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print("检索评测结果")
    print(f"Recall@5：  {metrics['recall_at_5']:.1%}")
    print(f"Recall@10： {metrics['recall_at_10']:.1%}")
    print(f"MRR：       {metrics['mrr']:.3f}")
    print(f"命中率：    {metrics['hit_rate']:.1%} ({int(metrics['hit_rate'] * metrics['question_count'])}/{int(metrics['question_count'])})")
    print(f"明细 CSV：  {results_path}")
    print(f"指标 CSV：  {metrics_path}")
    print(f"未命中：    {misses_path} ({len(misses)} 题)")
    print("=" * 70)

    if misses:
        print("\n未命中样例（前 5 题）：")
        for row in misses[:5]:
            print(f"\n[{row['question_id']}] {row['query']}")
            print(f"  期望：{row['gold_answer']}")
            print(f"  Top1：{row['top1_display_name']} | {row['top1_section']} | {row['top1_chunk_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索离线评测")
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help="评测题 JSONL 路径",
    )
    parser.add_argument("--top-k", type=int, default=10, help="检索 Top-K（用于 Recall@5/10）")
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
        print("\n[dry-run] 跳过 Milvus 检索。完成 embed 后请运行：")
        print("  python src/eval_retrieval.py")
        return

    results, metrics = run_retrieval_eval(questions, top_k=args.top_k)
    save_reports(results, metrics, top_k=args.top_k)


if __name__ == "__main__":
    main()
