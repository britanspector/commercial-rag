"""
RAG 问答 CLI：统一 Pipeline（混合召回 → Rerank → 带引用回答 / 低分拒答）。

用法：
    python src/rag_chat.py "京仪装备2026E毛利率预测是多少？"
    python src/rag_chat.py   # 交互模式
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from hf_env import bootstrap_hf_cache

bootstrap_hf_cache()


def _ensure_runtime_deps() -> None:
    """检测核心依赖；本项目默认使用 conda base 环境。"""
    missing: list[str] = []
    for mod in ("jieba", "pandas", "torch", "sentence_transformers", "pymilvus"):
        try:
            __import__(mod)
        except ModuleNotFoundError:
            missing.append(mod)
    if not missing:
        return
    print(f"缺少依赖：{', '.join(missing)}", file=sys.stderr)
    print(f"当前 Python：{sys.executable}", file=sys.stderr)
    if "commercial-rag" in sys.executable:
        print("本项目依赖安装在 base 环境，请执行：conda activate base", file=sys.stderr)
    else:
        print("请安装依赖：pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1)


_ensure_runtime_deps()

from rag_constants import DEFAULT_RERANK_REFUSAL_THRESHOLD, DEFAULT_RECALL_TOP_K, DEFAULT_RERANK_TOP_K
from rag_pipeline import RAGPipeline
from rag_types import RAGPipelineResult
from retrieval import RecallRoute


def print_answer(result: RAGPipelineResult) -> None:
    print("\n" + "=" * 70)
    if result.refused:
        msg = ""
        if result.evidence_check and result.evidence_check.refusal_message:
            msg = result.evidence_check.refusal_message
        print(
            f"[拒答] {result.refusal_reason} | top_rerank={result.top_rerank_score:.3f}"
            + (f"\n  {msg}" if msg else "")
        )
    else:
        print(f"[回答] top_rerank={result.top_rerank_score:.3f}")
    print("-" * 70)
    print(result.answer)
    if result.rerank_hits:
        print("-" * 70)
        print(f"重排 Top-{len(result.rerank_hits)} 片段：")
        for chunk in result.rerank_hits:
            page = ""
            if chunk.page_start:
                page = f" p.{chunk.page_start}" if chunk.page_start == chunk.page_end else (
                    f" p.{chunk.page_start}-{chunk.page_end}"
                )
            score = chunk.score_rerank if chunk.score_rerank is not None else chunk.score
            print(
                f"  #{chunk.rank} {chunk.company_name} — {chunk.section_title}{page} "
                f"(rerank={score:.3f})"
            )
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="研报 RAG 问答（引用溯源 + 拒答）")
    parser.add_argument("query", nargs="?", help="问题（省略则进入交互）")
    parser.add_argument("--recall-top-k", type=int, default=DEFAULT_RECALL_TOP_K)
    parser.add_argument("--rerank-top-k", type=int, default=DEFAULT_RERANK_TOP_K)
    parser.add_argument(
        "--refusal-threshold",
        type=float,
        default=DEFAULT_RERANK_REFUSAL_THRESHOLD,
    )
    parser.add_argument(
        "--recall-route",
        choices=[route.value for route in RecallRoute],
        default=RecallRoute.HYBRID.value,
        help="召回路线（默认 hybrid，与离线评测一致）",
    )
    args = parser.parse_args()

    pipeline = RAGPipeline(
        recall_top_k=args.recall_top_k,
        rerank_top_k=args.rerank_top_k,
        refusal_threshold=args.refusal_threshold,
        recall_route=args.recall_route,
    )

    try:
        if args.query:
            print_answer(pipeline.run(args.query))
            return

        print("研报 RAG 问答（输入空行或 quit 退出）")
        while True:
            try:
                query = input("\n问题> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query or query.lower() in {"quit", "exit", "q"}:
                break
            print_answer(pipeline.run(query))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
