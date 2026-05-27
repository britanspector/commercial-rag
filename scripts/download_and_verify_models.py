"""
下载项目核心模型并验证代码可从本地缓存读取。

用法：
    python3 scripts/download_and_verify_models.py

可选环境变量：
    HF_HOME=/root/autodl-tmp/hf_cache
    HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODELS = [
    "BAAI/bge-large-zh-v1.5",
    "BAAI/bge-reranker-v2-m3",
]


def download_models() -> dict[str, str]:
    print("=" * 70)
    print("开始下载/检查模型缓存")
    print("=" * 70)
    local_paths: dict[str, str] = {}
    for model_name in MODELS:
        print(f"\n[下载] {model_name}")
        local_path = snapshot_download(
            repo_id=model_name,
            repo_type="model",
            resume_download=True,
        )
        local_paths[model_name] = local_path
        print(f"[完成] {model_name} -> {local_path}")
    return local_paths


def verify_offline_loading() -> None:
    print("\n" + "=" * 70)
    print("开始离线加载验证（HF_HUB_OFFLINE=1）")
    print("=" * 70)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from chunk_mineru import get_tokenizer
    from embed_chunks import load_embedder
    from reranker import BGEReranker

    tokenizer = get_tokenizer()
    tokens = tokenizer.encode("测试模型缓存加载是否正常", add_special_tokens=False)
    print(f"[OK] tokenizer 加载成功，token 数: {len(tokens)}")

    embedder = load_embedder("cpu")
    emb = embedder.encode(
        ["这是一个embedding加载测试。"],
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    print(f"[OK] embedder 加载成功，向量维度: {emb.shape[-1]}")

    reranker = BGEReranker(device="cpu", use_fp16=False)
    scores = reranker.score_pairs(
        "公司2026年盈利预测如何？",
        ["公司预计2026年净利润同比增长。"],
        normalize=True,
    )
    print(f"[OK] reranker 加载成功，score: {scores[0]:.4f}")


def main() -> int:
    try:
        local_paths = download_models()
        verify_offline_loading()
    except Exception as error:
        print(f"\n[失败] {error}", file=sys.stderr)
        return 1

    print("\n" + "=" * 70)
    print("全部完成：模型已缓存且代码读取验证通过")
    for model_name, local_path in local_paths.items():
        print(f"- {model_name}: {local_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
