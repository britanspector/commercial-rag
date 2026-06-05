"""
答案生成 LLM 配置（Ollama 本地 qwen3:8b），与 RAGAS 评测配置分离。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_OLLAMA_BASE = "http://localhost:11434"
DEFAULT_LLM_MODEL = "qwen3:8b"


@dataclass
class GenerationConfig:
    llm_model: str
    ollama_base_url: str
    num_predict: int = 2048
    num_ctx: int = 8192
    ollama_disable_think: bool = True
    max_contexts: int = 3
    max_context_chars: int = 1200
    timeout_s: int = 180


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _ollama_host(base_url: str) -> str:
    host = (base_url or DEFAULT_OLLAMA_BASE).rstrip("/")
    if host.endswith("/v1"):
        host = host[:-3]
    return host or DEFAULT_OLLAMA_BASE


def resolve_generation_config() -> GenerationConfig:
    """
    环境变量：
        GEN_OLLAMA_BASE=http://localhost:11434
        GEN_LLM_MODEL=qwen3:8b
        GEN_NUM_PREDICT=2048
        GEN_NUM_CTX=8192
        GEN_OLLAMA_DISABLE_THINK=1
        GEN_MAX_CONTEXTS=3
        GEN_MAX_CONTEXT_CHARS=1200
        GEN_TIMEOUT_S=180
    """
    disable_think = _env("GEN_OLLAMA_DISABLE_THINK", "1").lower() not in ("0", "false", "no")
    return GenerationConfig(
        llm_model=_env("GEN_LLM_MODEL", DEFAULT_LLM_MODEL),
        ollama_base_url=_ollama_host(_env("GEN_OLLAMA_BASE", DEFAULT_OLLAMA_BASE)),
        num_predict=int(_env("GEN_NUM_PREDICT", "2048") or "2048"),
        num_ctx=int(_env("GEN_NUM_CTX", "8192") or "8192"),
        ollama_disable_think=disable_think,
        max_contexts=int(_env("GEN_MAX_CONTEXTS", "3") or "3"),
        max_context_chars=int(_env("GEN_MAX_CONTEXT_CHARS", "1200") or "1200"),
        timeout_s=int(_env("GEN_TIMEOUT_S", "180") or "180"),
    )


def describe_generation_config(cfg: GenerationConfig) -> str:
    think = ", think=off" if cfg.ollama_disable_think else ""
    return (
        f"gen_llm={cfg.llm_model} @ {cfg.ollama_base_url}, "
        f"num_predict={cfg.num_predict}, num_ctx={cfg.num_ctx}{think}, "
        f"contexts<={cfg.max_contexts}x{cfg.max_context_chars}chars, "
        f"timeout={cfg.timeout_s}s"
    )
