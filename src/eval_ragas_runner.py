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
from typing import Any, Literal

from eval_ragas_config import RagasRuntimeConfig, describe_config, resolve_ragas_config

RagasMetricsMode = Literal["all", "faith", "rel"]


class RagasChatOllama:
    """ChatOllama 子类：向 Ollama chat API 注入 think=False（Qwen3）。"""

    @staticmethod
    def create(cfg: RagasRuntimeConfig):
        from typing import AsyncIterator, Iterator, List, Mapping, Optional, Union

        from langchain_core.messages import BaseMessage
        from langchain_ollama import ChatOllama
        from ollama import Options

        model_lower = cfg.llm_model.lower()
        disable_think = cfg.ollama_disable_think and (
            "qwen3" in model_lower or "qwen-3" in model_lower
        )

        class _ChatOllama(ChatOllama):
            _ragas_disable_think: bool = disable_think

            def _think_kwargs(self) -> dict[str, bool]:
                return {"think": False} if self._ragas_disable_think else {}

            def _build_chat_params(
                self,
                messages: List[BaseMessage],
                stop: Optional[List[str]] = None,
                **kwargs: Any,
            ) -> tuple[list, dict[str, Any]]:
                ollama_messages = self._convert_messages_to_ollama_messages(messages)
                stop = stop if stop is not None else self.stop
                params = dict(self._default_params)
                for key in self._default_params:
                    if key in kwargs:
                        params[key] = kwargs[key]
                params["options"]["stop"] = stop
                return ollama_messages, params

            async def _acreate_chat_stream(
                self,
                messages: List[BaseMessage],
                stop: Optional[List[str]] = None,
                **kwargs: Any,
            ) -> AsyncIterator[Union[Mapping[str, Any], str]]:
                ollama_messages, params = self._build_chat_params(messages, stop, **kwargs)
                think_kw = self._think_kwargs()
                if "tools" in kwargs:
                    yield await self._async_client.chat(
                        model=params["model"],
                        messages=ollama_messages,
                        stream=False,
                        options=Options(**params["options"]),
                        keep_alive=params["keep_alive"],
                        format=params["format"],
                        tools=kwargs["tools"],
                        **think_kw,
                    )
                else:
                    async for part in await self._async_client.chat(
                        model=params["model"],
                        messages=ollama_messages,
                        stream=True,
                        options=Options(**params["options"]),
                        keep_alive=params["keep_alive"],
                        format=params["format"],
                        **think_kw,
                    ):
                        yield part

            def _create_chat_stream(
                self,
                messages: List[BaseMessage],
                stop: Optional[List[str]] = None,
                **kwargs: Any,
            ) -> Iterator[Union[Mapping[str, Any], str]]:
                ollama_messages, params = self._build_chat_params(messages, stop, **kwargs)
                think_kw = self._think_kwargs()
                if "tools" in kwargs:
                    yield self._client.chat(
                        model=params["model"],
                        messages=ollama_messages,
                        stream=False,
                        options=Options(**params["options"]),
                        keep_alive=params["keep_alive"],
                        format=params["format"],
                        tools=kwargs["tools"],
                        **think_kw,
                    )
                else:
                    yield from self._client.chat(
                        model=params["model"],
                        messages=ollama_messages,
                        stream=True,
                        options=Options(**params["options"]),
                        keep_alive=params["keep_alive"],
                        format=params["format"],
                        **think_kw,
                    )

        return _ChatOllama(
            model=cfg.llm_model,
            base_url=cfg.llm_base_url,
            temperature=0,
            num_predict=cfg.ollama_num_predict,
            num_ctx=cfg.ollama_num_ctx,
        )


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
    """启动前检查 LangChain 依赖版本。"""
    try:
        import langchain_core
        import langchain_ollama
        import langchain_openai

        core_ver = getattr(langchain_core, "__version__", "")
        openai_ver = getattr(langchain_openai, "__version__", "")
        ollama_ver = getattr(langchain_ollama, "__version__", "")
    except ImportError as exc:
        raise ImportError(
            "缺少 LangChain 依赖。请执行：pip install -r requirements-ragas.txt"
        ) from exc

    if core_ver.startswith("1.") or openai_ver.startswith("1."):
        raise ImportError(
            f"LangChain 版本不兼容 RAGAS 本地栈：langchain-core={core_ver}, "
            f"langchain-openai={openai_ver}, langchain-ollama={ollama_ver}。请执行：\n"
            "  pip install -r requirements-ragas.txt"
        )


def sample_input_stats(sample: dict) -> dict[str, Any]:
    contexts = sample.get("retrieved_contexts") or []
    return {
        "context_count": len(contexts),
        "context_total_chars": sum(len(str(c)) for c in contexts),
        "answer_chars": len(str(sample.get("response", ""))),
        "query": str(sample.get("user_input", ""))[:120],
    }


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
    # 优先用仅正文（不含【参考文献】元数据）的 answer_body 来做 RAGAS
    raw_answer = row.get("answer_body", None)
    if raw_answer is None:
        raw_answer = row.get("answer", "")
    answer = str(raw_answer)[: cfg.max_answer_chars]
    return {
        "question_id": row.get("question_id", ""),
        "user_input": str(row.get("query", "")),
        "response": answer,
        "retrieved_contexts": trimmed_ctx,
        "reference": str(row.get("gold_answer", "")),
    }


def _extract_finish_reason_and_content(resp) -> tuple[str | None, str]:
    """兼容 OpenAI finish_reason 与 Ollama done_reason。"""
    from langchain_core.outputs import ChatGeneration

    finish_reason = None
    content = ""
    if isinstance(resp, ChatGeneration) and resp.message is not None:
        meta = resp.message.response_metadata or {}
        finish_reason = meta.get("finish_reason") or meta.get("done_reason")
        content = resp.message.content or ""
    if resp.generation_info is not None:
        if finish_reason is None:
            finish_reason = resp.generation_info.get("finish_reason") or resp.generation_info.get(
                "done_reason"
            )
        if not content and hasattr(resp, "text"):
            content = resp.text or ""
    return finish_reason, content


def _ollama_is_finished(response) -> bool:
    """Ollama 用 done_reason=length 表示触顶；有内容时视为可用。"""
    for generation in response.flatten():
        resp = generation.generations[0][0]
        finish_reason, content = _extract_finish_reason_and_content(resp)
        if finish_reason in ("stop", "STOP", "MAX_TOKENS", "eos_token", None):
            continue
        if finish_reason == "length":
            if not str(content).strip():
                return False
            continue
        return False
    return True


def build_run_config(cfg: RagasRuntimeConfig):
    from ragas.run_config import RunConfig

    return RunConfig(
        timeout=cfg.run_timeout,
        max_retries=cfg.run_max_retries,
        max_workers=cfg.run_max_workers,
    )


def build_ragas_llm(cfg: RagasRuntimeConfig):
    try:
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise ImportError("请安装：pip install ragas langchain-ollama langchain-openai") from exc

    run_config = build_run_config(cfg)
    wrapper_kwargs: dict[str, Any] = {"run_config": run_config}

    if cfg.backend == "ollama":
        chat = RagasChatOllama.create(cfg)
        wrapper_kwargs["is_finished_parser"] = _ollama_is_finished
        return LangchainLLMWrapper(chat, **wrapper_kwargs)

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError("请安装：pip install langchain-openai") from exc

    kwargs: dict[str, Any] = {
        "model": cfg.llm_model,
        "api_key": cfg.llm_api_key,
        "temperature": 0,
        "timeout": cfg.ollama_timeout_s,
        "max_tokens": cfg.llm_max_tokens,
    }
    if cfg.llm_base_url:
        kwargs["base_url"] = cfg.llm_base_url
    return LangchainLLMWrapper(ChatOpenAI(**kwargs), **wrapper_kwargs)


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
    device = "cpu"
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS},
        )
    )


def _normalize_metrics_mode(metrics: str | None) -> RagasMetricsMode:
    if not metrics or metrics in ("all", "both"):
        return "all"
    normalized = metrics.lower().replace("faithfulness", "faith").replace("answer_relevancy", "rel")
    if normalized in ("faith", "rel"):
        return normalized  # type: ignore[return-value]
    return "all"


async def _score_one_sample(
    sample: dict,
    faith_metric,
    rel_metric,
    *,
    metrics_mode: RagasMetricsMode = "all",
) -> tuple[float | None, float | None, str]:
    from ragas.dataset_schema import SingleTurnSample

    turn = SingleTurnSample(
        user_input=sample["user_input"],
        response=sample["response"],
        retrieved_contexts=sample["retrieved_contexts"],
        reference=sample.get("reference") or "",
    )
    faith_score: float | None = None
    rel_score: float | None = None
    errors: list[str] = []

    if metrics_mode in ("all", "faith"):
        try:
            faith_score = float(await faith_metric.single_turn_ascore(turn))
        except Exception as exc:
            errors.append(f"faith:{exc}")

    if metrics_mode in ("all", "rel"):
        try:
            rel_score = float(await rel_metric.single_turn_ascore(turn))
        except Exception as exc:
            errors.append(f"rel:{exc}")

    return faith_score, rel_score, "; ".join(errors)[:500]


def _log_failure(row: dict, sample: dict, error: str) -> None:
    stats = sample_input_stats(sample)
    print(
        f"[RAGAS FAIL] {row.get('question_id')} "
        f"query={stats['query']!r} "
        f"contexts={stats['context_count']} "
        f"ctx_total={stats['context_total_chars']}chars "
        f"answer={stats['answer_chars']}chars | {error[:200]}",
        flush=True,
    )


async def _run_ragas_async(
    rows: list[dict],
    cfg: RagasRuntimeConfig,
    *,
    metrics_mode: RagasMetricsMode = "all",
) -> list[dict]:
    from ragas.metrics import AnswerRelevancy, Faithfulness

    llm = build_ragas_llm(cfg)
    embeddings = build_ragas_embeddings(cfg)
    faith_metric = Faithfulness(llm=llm)
    rel_metric = AnswerRelevancy(llm=llm, embeddings=embeddings)

    pending: list[tuple[dict, dict]] = []
    for row in rows:
        sample = prepare_ragas_sample(row, cfg)
        if sample is None:
            row["ragas_faithfulness"] = None
            row["ragas_answer_relevancy"] = None
            row["ragas_skipped"] = True
            continue
        faith = row.get("ragas_faithfulness")
        if faith is not None and faith == faith and metrics_mode == "all":
            continue
        pending.append((row, sample))

    total = len(pending)
    for index, (row, sample) in enumerate(pending, start=1):
        t0 = time.perf_counter()
        faith, rel, error = await _score_one_sample(
            sample, faith_metric, rel_metric, metrics_mode=metrics_mode
        )
        if metrics_mode in ("all", "faith"):
            row["ragas_faithfulness"] = faith
        if metrics_mode in ("all", "rel"):
            row["ragas_answer_relevancy"] = rel
        row["ragas_skipped"] = False
        row["ragas_error"] = error
        elapsed = time.perf_counter() - t0
        faith_v = row.get("ragas_faithfulness")
        rel_v = row.get("ragas_answer_relevancy")
        faith_s = f"{faith_v:.3f}" if faith_v is not None else "err"
        rel_s = f"{rel_v:.3f}" if rel_v is not None else "err"
        err_hint = f" | {error[:160]}" if error else ""
        print(
            f"[RAGAS {index}/{total}] {row.get('question_id')} "
            f"faith={faith_s} rel={rel_s} ({elapsed:.1f}s){err_hint}",
            flush=True,
        )
        if error:
            _log_failure(row, sample, error)

    return rows


def run_ragas_on_rows(
    rows: list[dict],
    *,
    backend: str | None = None,
    llm_model: str | None = None,
    metrics: str | None = None,
) -> tuple[dict[str, float], list[dict]]:
    """对已有 pipeline 结果逐题运行 RAGAS，返回聚合分与更新后的 rows。"""
    cfg = resolve_ragas_config(backend=backend, llm_model=llm_model)
    metrics_mode = _normalize_metrics_mode(metrics)
    print(f"[RAGAS] {describe_config(cfg)}")
    if metrics_mode != "all":
        print(f"[RAGAS] 仅评测指标：{metrics_mode}")

    check_langchain_stack()
    if cfg.backend == "ollama":
        if not check_ollama_reachable(cfg.llm_base_url):
            raise RuntimeError(
                f"无法连接 Ollama：{cfg.llm_base_url}。请先执行：ollama serve"
            )
        ensure_ollama_model(cfg)

    started = time.perf_counter()
    rows = asyncio.run(_run_ragas_async(rows, cfg, metrics_mode=metrics_mode))
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


def filter_rows_for_eval(
    rows: list[dict],
    *,
    question_id: str | None = None,
    index: int | None = None,
) -> list[dict]:
    """按 question_id 或 1-based index 筛选单条样本（调试用）。"""
    if question_id:
        matched = [r for r in rows if r.get("question_id") == question_id]
        if not matched:
            raise ValueError(f"未找到 question_id={question_id!r}")
        return matched
    if index is not None:
        if index < 1 or index > len(rows):
            raise ValueError(f"index 超出范围：{index}（共 {len(rows)} 题）")
        return [rows[index - 1]]
    return rows


def ensure_stdout_unbuffered() -> None:
    """避免长时间无进度输出（Windows 常见）。"""
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
