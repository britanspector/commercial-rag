"""语义缓存评测：公共类型、指标聚合、配置切换。"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval_generation_common import aggregate_generation_metrics, row_from_pipeline
from eval_retrieval import EvalQuestion, aggregate_metrics, is_hit_relevant, mrr, recall_at_k
from rag_types import RAGPipelineResult, RAGSearchResult

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "eval"


@dataclass
class CacheEvalMode:
    name: str
    enabled: bool
    l1_backend: str = "memory"
    l2_backend: str = "null"
    redis_url: str = ""

    @classmethod
    def off(cls) -> CacheEvalMode:
        return cls(name="cache_off", enabled=False)

    @classmethod
    def l1_only(cls, *, l1_backend: str = "memory", redis_url: str = "") -> CacheEvalMode:
        return cls(
            name="l1_only",
            enabled=True,
            l1_backend=l1_backend,
            l2_backend="null",
            redis_url=redis_url,
        )

    @classmethod
    def l1_l2(cls, *, l1_backend: str = "memory", redis_url: str = "") -> CacheEvalMode:
        return cls(
            name="l1_l2",
            enabled=True,
            l1_backend=l1_backend,
            l2_backend="milvus",
            redis_url=redis_url,
        )


def paraphrase_for_l2(question: EvalQuestion) -> str:
    """生成语义相近问法，用于 L2 命中测试（与 L1 精确 key 不同）。"""
    from cache_hit_paraphrase import make_paraphrase

    if question.query_type == "comparative":
        return f"帮我分析对比一下：{question.query.rstrip('？?').strip()}"
    if question.query_type == "summary":
        return make_paraphrase(question.query, "polite_tell")
    return make_paraphrase(question.query, "synonym_metric")


def apply_cache_mode(mode: CacheEvalMode, *, tmp_dir: Path) -> None:
    """通过环境变量配置缓存模式。"""
    os.environ["RAG_SEMANTIC_CACHE_ENABLED"] = "1" if mode.enabled else "0"
    os.environ["RAG_SEMANTIC_CACHE_BYPASS"] = "0"
    os.environ["RAG_SEMANTIC_CACHE_L1_BACKEND"] = mode.l1_backend
    os.environ["RAG_SEMANTIC_CACHE_L2_BACKEND"] = mode.l2_backend
    if mode.l2_backend == "milvus":
        os.environ["RAG_SEMANTIC_CACHE_SIM_THRESHOLD"] = "0.88"
    if mode.redis_url:
        os.environ["RAG_SEMANTIC_CACHE_REDIS_URL"] = mode.redis_url
    l2_db = tmp_dir / f"semantic_cache_{mode.name}.db"
    os.environ["RAG_SEMANTIC_CACHE_L2_MILVUS_DB"] = str(l2_db)


def reset_cache_runtime() -> None:
    from cache import reset_cache_manager
    from cache.config import load_cache_settings
    import cache.config as cache_config

    cache_config.cache_settings = load_cache_settings()
    reset_cache_manager()
    try:
        from cache.invalidate_hooks import invalidate_all_caches

        invalidate_all_caches()
    except Exception:
        pass


def prepare_isolated_cache_dir(tmp_dir: Path) -> None:
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)


def _retrieval_scores(hits: list[dict], question: EvalQuestion) -> dict[str, float]:
    relevant_ranks = [
        rank for rank, hit in enumerate(hits, start=1) if is_hit_relevant(hit, question)
    ]
    return {
        "recall_at_5": recall_at_k(relevant_ranks, 5),
        "recall_at_10": recall_at_k(relevant_ranks, 10),
        "mrr": mrr(relevant_ranks),
    }


def row_from_result(
    question: EvalQuestion,
    result: RAGPipelineResult | RAGSearchResult,
    *,
    phase: str,
    mode: str,
    elapsed_ms: float,
    paraphrase: bool = False,
) -> dict[str, Any]:
    if isinstance(result, RAGSearchResult):
        scores = _retrieval_scores([c.to_dict() for c in result.rerank_hits], question)
        row = {
            "question_id": question.id,
            "query_type": question.query_type,
            "query": question.query if not paraphrase else paraphrase_for_l2(question),
            "phase": phase,
            "mode": mode,
            "paraphrase": paraphrase,
            "top_rerank_score": result.top_rerank_score,
            "retrieval_hit": scores["recall_at_10"] > 0,
            **scores,
        }
    else:
        base = row_from_pipeline(question, result)
        row = dict(base)
        row["phase"] = phase
        row["mode"] = mode
        row["paraphrase"] = paraphrase
        scores = _retrieval_scores([c.to_dict() for c in result.rerank_hits], question)
        row.update(scores)

    cache = result.cache
    row.update(
        {
            "latency_ms": round(elapsed_ms, 2),
            "cache_hit": cache.hit if cache else False,
            "cache_source": cache.source if cache else "none",
            "cache_reason": cache.reason if cache else "",
            "cache_similarity": cache.similarity if cache else None,
            "safety_ok": cache.safety_ok if cache else True,
            "safety_reason": cache.safety_reason if cache else "",
            "lookup_ms": cache.lookup_ms if cache else 0.0,
            "pipeline_ms": cache.pipeline_ms if cache else elapsed_ms,
            "vector_retrieval": cache.vector_retrieval if cache else True,
            "llm_called": cache.llm_called if cache else True,
        }
    )
    return row


def aggregate_cache_eval_rows(rows: list[dict]) -> dict[str, Any]:
    if not rows:
        return {"question_count": 0}

    n = len(rows)
    hits = [r for r in rows if r.get("cache_hit")]
    gen_metrics = aggregate_generation_metrics(rows) if rows[0].get("answer") is not None else {}

    def _mean(key: str, subset: list[dict] | None = None) -> float:
        data = subset if subset is not None else rows
        vals = [float(r[key]) for r in data if r.get(key) is not None and r[key] == r[key]]
        return sum(vals) / len(vals) if vals else float("nan")

    retrieval_rows = rows
    retrieval_metrics = {}
    if retrieval_rows and "recall_at_3" in retrieval_rows[0]:
        retrieval_metrics = aggregate_metrics(retrieval_rows)

    return {
        "question_count": n,
        "cache_hit_rate": len(hits) / n if n else 0.0,
        "l1_hit_rate": sum(1 for r in rows if r.get("cache_source") == "l1_exact") / n if n else 0.0,
        "l2_hit_rate": sum(1 for r in rows if r.get("cache_source") == "l2_semantic") / n if n else 0.0,
        "avg_latency_ms": _mean("latency_ms"),
        "p50_latency_ms": _percentile([r["latency_ms"] for r in rows], 50),
        "p95_latency_ms": _percentile([r["latency_ms"] for r in rows], 95),
        "avg_lookup_ms": _mean("lookup_ms"),
        "avg_pipeline_ms": _mean("pipeline_ms", [r for r in rows if not r.get("cache_hit")]),
        "vector_retrieval_count": sum(1 for r in rows if r.get("vector_retrieval")),
        "llm_call_count": sum(1 for r in rows if r.get("llm_called")),
        "vector_retrievals_saved": sum(1 for r in rows if not r.get("vector_retrieval")),
        "llm_calls_saved": sum(1 for r in rows if not r.get("llm_called")),
        "safety_reject_count": sum(1 for r in rows if not r.get("safety_ok")),
        "recall_at_5": _mean("recall_at_5"),
        "recall_at_10": _mean("recall_at_10"),
        "mrr": _mean("mrr"),
        **{
            k: v
            for k, v in gen_metrics.items()
            if k
            in {
                "refusal_accuracy",
                "citation_accuracy",
                "faithfulness_ragas",
                "answer_relevancy_ragas",
                "retrieval_hit_rate",
                "answer_factually_supported_rate",
            }
        },
        **{
            f"retrieval_{k}": v
            for k, v in retrieval_metrics.items()
            if k in {"recall_at_5", "recall_at_10", "mrr", "hit_rate"}
        },
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(len(ordered) * pct / 100) - 1))
    return round(ordered[idx], 2)


@dataclass
class CriteriaCheck:
    id: int
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CriteriaReport:
    checks: list[CriteriaCheck] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_markdown(self) -> str:
        lines = ["| # | 完成标准 | 结果 | 说明 |", "|---|---------|------|------|"]
        for check in self.checks:
            status = "通过" if check.passed else "未通过"
            lines.append(f"| {check.id} | {check.name} | {status} | {check.detail} |")
        return "\n".join(lines)
