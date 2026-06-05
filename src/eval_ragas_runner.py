"""
RAGAS 执行器：Ollama Qwen3-8B + 本地 bge Embedding，逐题打分并写回 rows。
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from eval_ragas_config import RagasRuntimeConfig, describe_config, resolve_ragas_config


def check_ollama_reachable(base_url: str, timeout: float = 3.0) -> bool:
    root = base_url.rstrip("/").replace("/v1", "")
    try:
        with urllib.request.urlopen(f"{root}/api/tags", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def list_ollama_models(base_url: str, timeout: float = 5.0) -> list[str]:
    root = base_url.rstrip("/").replace("/v1", "")
    try:
        with urllib.request.urlopen(f"{root}/api/tags", timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return []
    names: list[str] = []
    for item in payload.get("models", []):
        name = item.get("name", "")
        if name:
            names.append(name)
    return names


def ensure_ollama_model(cfg: RagasRuntimeConfig) -> None:
    """确认 Ollama 已拉取目标模型，避免 404 model not found。"""
    models = list_ollama_models(cfg.llm_base_url)
    if not models:
        return
    target = cfg.llm_model
    candidates = {target, f"{target}:latest", target.replace(":", "-")}
    if not any(name in candidates or name.split(":")[0] == target.split(":")[0] for name in models):
        sample = ", ".join(models[:8])
        raise RuntimeError(
            f"Ollama 未找到模型 '{target}'。请先执行：ollama pull {target}\n"
            f"当前已安装：{sample}{'...' if len(models) > 8 else ''}"
        )


def check_langchain_stack() -> None:
    """启动前检查 LangChain 0.2.x 成套版本，避免混装 1.x 导致 ImportError。"""
    try:
        import langchain_core
        import langchain_openai

        core_ver = getattr(langchain_core, "__version__", "")
        openai_ver = getattr(langchain_openai, "__version__", "")
    except ImportError as exc:
        raise ImportError(
            "缺少 LangChain 依赖。请执行：pip install -r requirements-ragas.txt"
        ) from exc

    if core_ver.startswith("1.") or openai_ver.startswith("1."):
        raise ImportError(
            f"LangChain 版本不兼容 RAGAS 本地栈：langchain-core={core_ver}, "
            f"langchain-openai={openai_ver}。请执行：\n"
            "  pip install langchain-openai==0.1.25 langchain-core==0.2.43 "
            "langchain-community==0.2.19"
        )


def prepare_ragas_sample(row: dict, cfg: RagasRuntimeConfig) -> dict[str, Any] | None:
    if row.get("refused"):
        return None
    contexts = row.get("ragas_contexts") or []
    if isinstance(contexts, str):
        try:
            contexts = json.loads(contexts)
        except json.JSONDecodeError:
            contexts = []
    if not contexts:
        return None

    trimmed_ctx = [str(c)[: cfg.max_context_chars] for c in contexts[: cfg.max_contexts]]
    answer = str(row.get("answer", ""))[: cfg.max_answer_chars]
    return {
        "question_id": row.get("question_id", ""),
        "user_input": str(row.get("query", "")),
        "response": answer,
        "retrieved_contexts": trimmed_ctx,
        "reference": str(row.get("gold_answer", "")),
    }


def build_ragas_llm(cfg: RagasRuntimeConfig):
    try:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise ImportError(
            "请安装：pip install ragas langchain-openai langchain-community"
        ) from exc

    kwargs: dict[str, Any] = {
        "model": cfg.llm_model,
        "api_key": cfg.llm_api_key,
        "temperature": 0,
        "timeout": cfg.ollama_timeout_s,
    }
    if cfg.llm_base_url:
        kwargs["base_url"] = cfg.llm_base_url
    return LangchainLLMWrapper(ChatOpenAI(**kwargs))


def build_ragas_embeddings(cfg: RagasRuntimeConfig):
    if cfg.embed_backend == "bge_local":
        return _build_bge_embeddings()

    try:
        from langchain_openai import OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as exc:
        raise ImportError("请安装：pip install langchain-openai") from exc

    kwargs: dict[str, Any] = {
        "model": cfg.embed_model,
        "api_key": cfg.embed_api_key,
    }
    if cfg.embed_base_url:
        kwargs["base_url"] = cfg.embed_base_url
    return LangchainEmbeddingsWrapper(OpenAIEmbeddings(**kwargs))


def _build_bge_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    from embed_chunks import EMBED_MODEL, NORMALIZE_EMBEDDINGS
    from hf_env import resolve_local_model_path

    model_path = resolve_local_model_path(EMBED_MODEL)
    # RAGAS 阶段默认 CPU，把 GPU 留给 Ollama（4060 8GB 笔记本）
    device = "cpu"
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS},
        )
    )


async def _score_one_sample(sample: dict, faith_metric, rel_metric) -> tuple[float | None, float | None]:
    from ragas.dataset_schema import SingleTurnSample

    turn = SingleTurnSample(
        user_input=sample["user_input"],
        response=sample["response"],
        retrieved_contexts=sample["retrieved_contexts"],
        reference=sample.get("reference") or "",
    )
    faith_score = await faith_metric.single_turn_ascore(turn)
    rel_score = await rel_metric.single_turn_ascore(turn)
    return float(faith_score), float(rel_score)


async def _run_ragas_async(
    rows: list[dict],
    cfg: RagasRuntimeConfig,
) -> list[dict]:
    from ragas.metrics import AnswerRelevancy, Faithfulness

    llm = build_ragas_llm(cfg)
    embeddings = build_ragas_embeddings(cfg)
    faith_metric = Faithfulness(llm=llm)
    rel_metric = AnswerRelevancy(llm=llm, embeddings=embeddings)

    scored: list[dict] = []
    pending: list[tuple[dict, dict]] = []
    for row in rows:
        sample = prepare_ragas_sample(row, cfg)
        if sample is None:
            row["ragas_faithfulness"] = None
            row["ragas_answer_relevancy"] = None
            row["ragas_skipped"] = True
            continue
        faith = row.get("ragas_faithfulness")
        if faith is not None and faith == faith:
            continue
        pending.append((row, sample))

    total = len(pending)
    for index, (row, sample) in enumerate(pending, start=1):
        t0 = time.perf_counter()
        try:
            faith, rel = await _score_one_sample(sample, faith_metric, rel_metric)
            row["ragas_faithfulness"] = faith
            row["ragas_answer_relevancy"] = rel
            row["ragas_skipped"] = False
            row["ragas_error"] = ""
        except Exception as exc:
            row["ragas_faithfulness"] = None
            row["ragas_answer_relevancy"] = None
            row["ragas_skipped"] = False
            row["ragas_error"] = str(exc)[:500]
        elapsed = time.perf_counter() - t0
        scored.append(row)
        faith_v = row.get("ragas_faithfulness")
        rel_v = row.get("ragas_answer_relevancy")
        faith_s = f"{faith_v:.3f}" if faith_v is not None else "err"
        rel_s = f"{rel_v:.3f}" if rel_v is not None else "err"
        err_hint = ""
        if row.get("ragas_error"):
            err_hint = f" | {row['ragas_error'][:160]}"
        print(
            f"[RAGAS {index}/{total}] {row.get('question_id')} "
            f"faith={faith_s} rel={rel_s} ({elapsed:.1f}s){err_hint}",
            flush=True,
        )

    return rows


def run_ragas_on_rows(
    rows: list[dict],
    *,
    backend: str | None = None,
    llm_model: str | None = None,
) -> tuple[dict[str, float], list[dict]]:
    """对已有 pipeline 结果逐题运行 RAGAS，返回聚合分与更新后的 rows。"""
    cfg = resolve_ragas_config(backend=backend, llm_model=llm_model)
    print(f"[RAGAS] {describe_config(cfg)}")

    check_langchain_stack()
    if cfg.backend == "ollama":
        if not check_ollama_reachable(cfg.llm_base_url):
            raise RuntimeError(
                f"无法连接 Ollama：{cfg.llm_base_url}。请先执行：ollama serve"
            )
        ensure_ollama_model(cfg)

    started = time.perf_counter()
    rows = asyncio.run(_run_ragas_async(rows, cfg))
    print(f"[RAGAS] 完成，耗时 {time.perf_counter() - started:.1f}s")

    scored = [
        r
        for r in rows
        if not r.get("refused")
        and r.get("ragas_faithfulness") is not None
        and r.get("ragas_faithfulness") == r.get("ragas_faithfulness")
    ]

    def _mean(key: str) -> float:
        vals = [float(r[key]) for r in scored if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    summary = {
        "faithfulness": _mean("ragas_faithfulness"),
        "answer_relevancy": _mean("ragas_answer_relevancy"),
        "ragas_scored_n": len(scored),
    }
    return summary, rows


def load_rows_from_detail(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_rows_from_results_csv(path: Path) -> list[dict]:
    import pandas as pd

    df = pd.read_csv(path)
    rows = df.to_dict(orient="records")
    for row in rows:
        for key in ("citations_json", "evidence_check_json", "ragas_contexts"):
            if key in row and isinstance(row[key], str):
                try:
                    row[key] = json.loads(row[key])
                except json.JSONDecodeError:
                    pass
    return rows


def load_rows_for_ragas(
    detail_path: Path | None,
    results_path: Path | None,
) -> list[dict]:
    if detail_path and detail_path.exists():
        return load_rows_from_detail(detail_path)
    if results_path and results_path.exists():
        rows = load_rows_from_results_csv(results_path)
        if rows and "ragas_contexts" not in rows[0]:
            raise ValueError(
                "CSV 缺少 ragas_contexts，请使用 --save-detail 生成的 eval_generation_detail.jsonl"
            )
        return rows
    raise FileNotFoundError("未找到 detail.jsonl 或 results.csv，请先运行 eval_generation.py --save-detail")


def ensure_stdout_unbuffered() -> None:
    """避免长时间无进度输出（Windows 常见）。"""
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
