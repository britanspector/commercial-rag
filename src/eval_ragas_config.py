"""
RAGAS 评判后端配置：Ollama 本地 Qwen、OpenAI 兼容 API、本地 bge Embedding。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

RagasBackend = Literal["ollama", "openai", "auto"]

DEFAULT_OLLAMA_BASE = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_API_KEY = "ollama"

DEFAULT_API_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_MODEL_API = "text-embedding-3-small"


@dataclass
class RagasRuntimeConfig:
    backend: RagasBackend
    llm_model: str
    llm_base_url: str
    llm_api_key: str
    embed_backend: Literal["bge_local", "openai"]
    embed_model: str
    embed_base_url: str
    embed_api_key: str
    max_contexts: int = 3
    max_context_chars: int = 1200
    max_answer_chars: int = 1200
    ollama_num_predict: int = 4096
    ollama_num_ctx: int = 8192
    ollama_disable_think: bool = True
    ollama_timeout_s: int = 300
    run_timeout: int = 180
    run_max_retries: int = 2
    run_max_workers: int = 1
    # OpenAI 兼容后端仍使用 max_tokens
    llm_max_tokens: int = 4096


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _ollama_host(base_url: str) -> str:
    """ChatOllama 使用 host（无 /v1 后缀）。"""
    host = (base_url or DEFAULT_OLLAMA_BASE).rstrip("/")
    if host.endswith("/v1"):
        host = host[:-3]
    return host or DEFAULT_OLLAMA_BASE


def resolve_ragas_config(
    *,
    backend: str | None = None,
    llm_model: str | None = None,
) -> RagasRuntimeConfig:
    """
    解析 RAGAS 运行时配置。

    环境变量（常用）：
        RAGAS_BACKEND=ollama|openai|auto
        RAGAS_LLM_MODEL=qwen3:8b
        RAGAS_OLLAMA_BASE=http://localhost:11434
        RAGAS_OLLAMA_NUM_PREDICT=4096
        RAGAS_OLLAMA_NUM_CTX=8192
        RAGAS_OLLAMA_DISABLE_THINK=1
        RAGAS_MAX_CONTEXT_CHARS=1200
        RAGAS_MAX_ANSWER_CHARS=1200
        RAGAS_RUN_TIMEOUT=180
        RAGAS_RUN_MAX_RETRIES=2
        RAGAS_RUN_MAX_WORKERS=1
        OPENAI_API_KEY / RAGAS_API_KEY（openai 后端）
    """
    chosen = (backend or _env("RAGAS_BACKEND", "auto")).lower()
    ollama_base = _ollama_host(_env("RAGAS_OLLAMA_BASE", DEFAULT_OLLAMA_BASE))
    ollama_model = llm_model or _env("RAGAS_LLM_MODEL", DEFAULT_OLLAMA_MODEL)
    ollama_key = _env("RAGAS_OLLAMA_API_KEY", DEFAULT_OLLAMA_API_KEY)

    api_key = _env("RAGAS_API_KEY") or _env("OPENAI_API_KEY")
    api_base = _env("RAGAS_OPENAI_BASE") or _env("OPENAI_API_BASE")
    api_model = llm_model or _env("RAGAS_LLM_MODEL", DEFAULT_API_MODEL)

    embed_backend = _env("RAGAS_EMBED_BACKEND", "bge_local").lower()
    if embed_backend not in ("bge_local", "openai"):
        embed_backend = "bge_local"

    if chosen == "auto":
        if ollama_base and ollama_model:
            chosen = "ollama"
        elif api_key:
            chosen = "openai"
        else:
            chosen = "ollama"

    llm_max_tokens = int(_env("RAGAS_LLM_MAX_TOKENS", "4096") or "4096")
    num_predict = int(_env("RAGAS_OLLAMA_NUM_PREDICT", "4096") or "4096")
    num_ctx = int(_env("RAGAS_OLLAMA_NUM_CTX", "8192") or "8192")
    disable_think = _env("RAGAS_OLLAMA_DISABLE_THINK", "1").lower() not in ("0", "false", "no")
    run_timeout = int(_env("RAGAS_RUN_TIMEOUT", "180") or "180")
    run_max_retries = int(_env("RAGAS_RUN_MAX_RETRIES", "2") or "2")
    run_max_workers = int(_env("RAGAS_RUN_MAX_WORKERS", "1") or "1")

    common = dict(
        max_contexts=int(_env("RAGAS_MAX_CONTEXTS", "3") or "3"),
        max_context_chars=int(_env("RAGAS_MAX_CONTEXT_CHARS", "1200") or "1200"),
        max_answer_chars=int(_env("RAGAS_MAX_ANSWER_CHARS", "1200") or "1200"),
        run_timeout=run_timeout,
        run_max_retries=run_max_retries,
        run_max_workers=run_max_workers,
        ollama_timeout_s=int(_env("RAGAS_OLLAMA_TIMEOUT", "300") or "300"),
    )

    if chosen == "ollama":
        return RagasRuntimeConfig(
            backend="ollama",
            llm_model=ollama_model,
            llm_base_url=ollama_base,
            llm_api_key=ollama_key or DEFAULT_OLLAMA_API_KEY,
            embed_backend=embed_backend,  # type: ignore[arg-type]
            embed_model=_env("RAGAS_EMBED_MODEL", DEFAULT_EMBED_MODEL_API),
            embed_base_url=api_base,
            embed_api_key=api_key or ollama_key,
            ollama_num_predict=num_predict,
            ollama_num_ctx=num_ctx,
            ollama_disable_think=disable_think,
            llm_max_tokens=llm_max_tokens,
            **common,
        )

    if not api_key:
        raise ValueError(
            "RAGAS openai 后端需要 OPENAI_API_KEY 或 RAGAS_API_KEY；"
            "本地请用 --ragas-backend ollama 并先 ollama pull qwen3:8b"
        )

    return RagasRuntimeConfig(
        backend="openai",
        llm_model=api_model,
        llm_base_url=api_base,
        llm_api_key=api_key,
        embed_backend=embed_backend,  # type: ignore[arg-type]
        embed_model=_env("RAGAS_EMBED_MODEL", DEFAULT_EMBED_MODEL_API),
        embed_base_url=api_base,
        embed_api_key=api_key,
        ollama_disable_think=False,
        llm_max_tokens=llm_max_tokens,
        **common,
    )


def describe_config(cfg: RagasRuntimeConfig) -> str:
    embed = "bge-large-zh (local)" if cfg.embed_backend == "bge_local" else cfg.embed_model
    if cfg.backend == "ollama":
        think = ", think=off" if cfg.ollama_disable_think else ""
        llm_part = (
            f"llm={cfg.llm_model} @ {cfg.llm_base_url}, "
            f"num_predict={cfg.ollama_num_predict}, num_ctx={cfg.ollama_num_ctx}{think}"
        )
    else:
        llm_part = f"llm={cfg.llm_model}, max_tokens={cfg.llm_max_tokens}"
    return (
        f"backend={cfg.backend}, {llm_part}, "
        f"timeout={cfg.run_timeout}s, max_retries={cfg.run_max_retries}, "
        f"max_workers={cfg.run_max_workers}, "
        f"contexts<={cfg.max_contexts}x{cfg.max_context_chars}chars, "
        f"answer<={cfg.max_answer_chars}chars, embed={embed}"
    )
