#!/usr/bin/env bash
# 使用 base 环境运行 RAG 问答（依赖与 HF 缓存均在此机 base 中）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/autodl-tmp/hf_cache/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/autodl-tmp/hf_cache/hub}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-/root/autodl-tmp/hf_cache/sentence_transformers}"

exec "$PYTHON" "$ROOT/src/rag_chat.py" "$@"
