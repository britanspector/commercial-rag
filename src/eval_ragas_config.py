"""
RAGAS 评判后端配置：Ollama 本地 Qwen、OpenAI 兼容 API、本地 bge Embedding。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

RagasBackend = Literal["ollama", "openai", "auto"]

DEFAULT_OLLAMA_BASE = "http://localhost:11434/v1"
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
    max_context_chars: int = 4000
    max_answer_chars: int = 3500
    ollama_timeout_s: int = 300


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def resolve_ragas_config(
    *,
    backend: str | None = None,
    llm_model: str | None = None,
) -> RagasRuntimeConfig:
    """
    解析 RAGAS 运行时配置。

    环境变量（常用）：
        RAGAS_BACKEND=ollama|openai|auto  （auto 且有 Ollama 配置时优先 ollama）
        RAGAS_LLM_MODEL=qwen3:8b
        RAGAS_OLLAMA_BASE=http://localhost:11434/v1
        RAGAS_OLLAMA_API_KEY=ollama
        RAGAS_EMBED_BACKEND=bge_local|openai
        OPENAI_API_KEY / RAGAS_API_KEY（openai 后端或 openai embedding）
    """
    chosen = (backend or _env("RAGAS_BACKEND", "auto")).lower()
    ollama_base = _env("RAGAS_OLLAMA_BASE", DEFAULT_OLLAMA_BASE)
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

    if chosen == "ollama":
        return RagasRuntimeConfig(
            backend="ollama",
            llm_model=ollama_model,
            llm_base_url=ollama_base or DEFAULT_OLLAMA_BASE,
            llm_api_key=ollama_key or DEFAULT_OLLAMA_API_KEY,
            embed_backend=embed_backend,  # type: ignore[arg-type]
            embed_model=_env("RAGAS_EMBED_MODEL", DEFAULT_EMBED_MODEL_API),
            embed_base_url=api_base,
            embed_api_key=api_key or ollama_key,
            max_contexts=int(_env("RAGAS_MAX_CONTEXTS", "3") or "3"),
            max_context_chars=int(_env("RAGAS_MAX_CONTEXT_CHARS", "4000") or "4000"),
            max_answer_chars=int(_env("RAGAS_MAX_ANSWER_CHARS", "3500") or "3500"),
            ollama_timeout_s=int(_env("RAGAS_OLLAMA_TIMEOUT", "300") or "300"),
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
        max_contexts=int(_env("RAGAS_MAX_CONTEXTS", "3") or "3"),
        max_context_chars=int(_env("RAGAS_MAX_CONTEXT_CHARS", "4000") or "4000"),
        max_answer_chars=int(_env("RAGAS_MAX_ANSWER_CHARS", "3500") or "3500"),
        ollama_timeout_s=int(_env("RAGAS_OLLAMA_TIMEOUT", "300") or "300"),
    )


def describe_config(cfg: RagasRuntimeConfig) -> str:
    embed = "bge-large-zh (local)" if cfg.embed_backend == "bge_local" else cfg.embed_model
    return (
        f"backend={cfg.backend}, llm={cfg.llm_model} @ {cfg.llm_base_url or 'default'}, "
        f"embed={embed}"
    )
