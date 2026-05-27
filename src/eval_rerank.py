"""
Rerank 离线评测：混合 Top5 vs 混合 Top20 → Rerank → Top5。

前置：
    python src/embed_chunks.py
    python src/build_bm25_index.py

用法：
    python src/eval_rerank.py
    python src/eval_rerank.py --skip-answer
"""

from __future__ import annotations

import argparse
import gc
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

PROJECT_ROOT = CURRENT_DIR.parent

from embed_chunks import EMBED_DIM, EMBED_MODEL, OUTPUT_MILVUS_DB, load_embedder, resolve_device
from bm25_store import DEFAULT_INDEX_PATH
from eval_retrieval import DEFAULT_QUESTIONS, encode_query, load_chunk_id_set, load_questions, validate_questions
from eval_rerank_common import FINAL_TOP_K, RECALL_POOL
from rag_constants import DEFAULT_RERANK_REFUSAL_THRESHOLD
from retrieval import HybridRetriever, RecallRoute, DEFAULT_HYBRID_VECTOR_WEIGHT


HYBRID_POOL_CACHE = Path(tempfile.gettempdir()) / "commercial_rag_hybrid_rerank_pools.pkl"


def run_eval(
    questions,
    skip_answer: bool,
    refusal_threshold: float,
    questions_path: Path = DEFAULT_QUESTIONS,
    hybrid_vector_weight: float = DEFAULT_HYBRID_VECTOR_WEIGHT,
) -> None:
    device = resolve_device()
    print(f"Embedding 模型：{EMBED_MODEL}")
    print(f"Reranker 模型：BAAI/bge-reranker-v2-m3")
    print(f"设备：{device}")
    print(f"初召回：混合（向量+BM25，权重 {hybrid_vector_weight}/{1 - hybrid_vector_weight:.1f}）")
    print(f"对比：混合直接 Top{FINAL_TOP_K} vs 混合 Top{RECALL_POOL} → Rerank → Top{FINAL_TOP_K}")
    print(f"拒答阈值（normalize rerank score）：{refusal_threshold}")

    if not OUTPUT_MILVUS_DB.exists():
        raise FileNotFoundError(f"请先运行 embed_chunks.py\n{OUTPUT_MILVUS_DB}")
    if not DEFAULT_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"请先运行 build_bm25_index.py\n{DEFAULT_INDEX_PATH}"
        )

    print("\n[1/2] 混合初召回 ...")
    embedder = load_embedder(device)
    retriever = HybridRetriever.from_paths(
        OUTPUT_MILVUS_DB,
        vector_dim=EMBED_DIM,
        hybrid_vector_weight=hybrid_vector_weight,
        hybrid_pool_size=RECALL_POOL,
    )
    retriever.milvus_store.load()

    pools = []
    for question in questions:
        query_vector = encode_query(embedder, question.query)
        pools.append(
            retriever.retrieve(
                RecallRoute.HYBRID,
                question.query,
                query_vector,
                RECALL_POOL,
            )
        )

    retriever.close()
    del embedder
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    with open(HYBRID_POOL_CACHE, "wb") as cache_file:
        pickle.dump(pools, cache_file)
    del pools
    gc.collect()

    questions_abs = questions_path.resolve()
    print("[2/2] 启动 Rerank 子进程 ...")
    subprocess.run(
        [
            sys.executable,
            str(CURRENT_DIR / "eval_rerank_phase2.py"),
            str(HYBRID_POOL_CACHE),
            str(questions_abs),
            "1" if skip_answer else "0",
            str(refusal_threshold),
        ],
        check=True,
    )
    try:
        HYBRID_POOL_CACHE.unlink(missing_ok=True)
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank 检索与生成评测")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--skip-answer", action="store_true")
    parser.add_argument(
        "--refusal-threshold",
        type=float,
        default=DEFAULT_RERANK_REFUSAL_THRESHOLD,
    )
    parser.add_argument(
        "--hybrid-vector-weight",
        type=float,
        default=DEFAULT_HYBRID_VECTOR_WEIGHT,
        help="混合召回中向量分权重（默认 0.5）",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    questions_path = args.questions.resolve()
    questions = load_questions(questions_path)
    validate_questions(questions, load_chunk_id_set())

    if args.dry_run:
        print("[dry-run] 跳过 Rerank 评测")
        return

    run_eval(
        questions,
        skip_answer=args.skip_answer,
        refusal_threshold=args.refusal_threshold,
        questions_path=questions_path,
        hybrid_vector_weight=args.hybrid_vector_weight,
    )


if __name__ == "__main__":
    main()
