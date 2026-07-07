#!/usr/bin/env python3
"""生成 cache_hit_pairs.jsonl（≥100 组 paraphrase 测试对）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from cache_hit_paraphrase import expected_layer, make_paraphrase
from eval_retrieval import load_questions

OUTPUT = CURRENT_DIR.parent / "data" / "eval" / "cache_hit_pairs.jsonl"
QUESTIONS_PATH = CURRENT_DIR.parent / "data" / "eval" / "eval_questions.jsonl"

REGRESSION_PAIRS = [
    {
        "pair_id": "regression_01",
        "seed_id": "user",
        "variant_type": "synonym_metric",
        "original_query": "澜起科技2024年的EPS是多少？",
        "paraphrase_query": "请告诉我2024年澜起科技的每股收益？",
        "stock_code": "",
        "query_type": "factual",
        "expected_layer": "l2_semantic",
        "notes": "用户报告 badcase：EPS↔每股收益 + 语序",
    },
    {
        "pair_id": "regression_02",
        "seed_id": "user",
        "variant_type": "punctuation",
        "original_query": "澜起科技2026年EPS是多少？",
        "paraphrase_query": "请问澜起科技2026年EPS是？",
        "stock_code": "",
        "query_type": "factual",
        "expected_layer": "l1_exact",
        "notes": "用户报告 badcase：礼貌前缀 + 句末差异",
    },
]

VARIANT_ROTATION = [
    "polite_prefix",
    "synonym_metric",
    "word_order",
    "punctuation",
    "filler_suffix",
    "polite_tell",
]


def main() -> None:
    questions = [
        q
        for q in load_questions(QUESTIONS_PATH)
        if q.query_type in ("factual", "summary")
    ]
    pairs: list[dict] = list(REGRESSION_PAIRS)
    pair_idx = len(pairs)

    for qi, question in enumerate(questions):
        variants = VARIANT_ROTATION[qi % len(VARIANT_ROTATION) : qi % len(VARIANT_ROTATION) + 2]
        if len(variants) < 2:
            variants = VARIANT_ROTATION[:2]
        for vi, variant_type in enumerate(variants):
            try:
                paraphrase = make_paraphrase(question.query, variant_type)
            except Exception:
                continue
            if normalize_equal(question.query, paraphrase) and variant_type != "punctuation":
                continue
            pair_idx += 1
            pairs.append(
                {
                    "pair_id": f"pair_{pair_idx:04d}",
                    "seed_id": question.id,
                    "variant_type": variant_type,
                    "original_query": question.query,
                    "paraphrase_query": paraphrase,
                    "stock_code": "",
                    "query_type": question.query_type,
                    "expected_layer": expected_layer(question.query, paraphrase),
                    "gold_stock_code": question.stock_code,
                }
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} pairs to {OUTPUT}")


def normalize_equal(a: str, b: str) -> bool:
    from cache.policy import normalize_query

    return normalize_query(a) == normalize_query(b)


if __name__ == "__main__":
    main()
