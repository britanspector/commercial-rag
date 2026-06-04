"""
Hugging Face 本地缓存与离线加载（AutoDL 等无法访问 huggingface.co 的环境）。

须在 import sentence_transformers / transformers 之前调用 bootstrap_hf_cache()。
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_HF_CACHE = Path("/root/autodl-tmp/hf_cache")

CORE_MODELS = (
    "BAAI/bge-large-zh-v1.5",
    "BAAI/bge-reranker-v2-m3",
)

_bootstrapped = False


def hf_hub_cache_root() -> Path:
    """从环境变量或默认 AutoDL 路径推导 HF hub 缓存根目录。"""
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(key, "").strip()
        if value:
            return Path(value)

    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        return Path(hf_home) / "hub"

    default_hub = DEFAULT_HF_CACHE / "hub"
    if default_hub.is_dir():
        return default_hub

    return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_local_model_path(model_name: str) -> str:
    """若本地 snapshot 存在则返回绝对路径，否则返回原始 model id。"""
    cache_root = hf_hub_cache_root()
    repo_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    refs_main = repo_dir / "refs" / "main"
    if not refs_main.exists():
        return model_name

    snapshot_id = refs_main.read_text(encoding="utf-8").strip()
    snapshot_dir = repo_dir / "snapshots" / snapshot_id
    if snapshot_dir.is_dir() and any(snapshot_dir.iterdir()):
        return str(snapshot_dir)
    return model_name


def is_model_cached(model_name: str) -> bool:
    return resolve_local_model_path(model_name) != model_name


def bootstrap_hf_cache() -> None:
    """设置 HF 缓存路径；核心模型齐全时启用离线模式，避免联网 HEAD 请求。"""
    global _bootstrapped
    if _bootstrapped:
        return
    _bootstrapped = True

    default_hub = DEFAULT_HF_CACHE / "hub"
    if default_hub.is_dir():
        os.environ.setdefault("HF_HOME", str(DEFAULT_HF_CACHE))
        os.environ.setdefault("HF_HUB_CACHE", str(default_hub))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(default_hub))
        os.environ.setdefault(
            "SENTENCE_TRANSFORMERS_HOME",
            str(DEFAULT_HF_CACHE / "sentence_transformers"),
        )

    if all(is_model_cached(model) for model in CORE_MODELS):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def local_files_only() -> bool:
    bootstrap_hf_cache()
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")
