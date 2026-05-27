#!/usr/bin/env bash
# commercial-rag → AutoDL 打包脚本（Linux / macOS / AutoDL 服务器本地）
#
# 用法:
#   bash scripts/pack_for_autodl.sh                      # 默认 tier=essential
#   bash scripts/pack_for_autodl.sh --tier minimal
#   bash scripts/pack_for_autodl.sh --tier full --out-dir /root/autodl-tmp
#   bash scripts/pack_for_autodl.sh --include-eval-artifacts
#
# 仅生成压缩包，不自动上传。

set -euo pipefail

TIER="essential"
OUT_DIR=""
INCLUDE_EVAL_ARTIFACTS=0
INCLUDE_NOTES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier) TIER="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --include-eval-artifacts) INCLUDE_EVAL_ARTIFACTS=1; shift ;;
    --include-notes) INCLUDE_NOTES=1; shift ;;
    -h|--help)
      echo "Usage: bash scripts/pack_for_autodl.sh [--tier minimal|essential|recommended|full] [--out-dir DIR]"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$PROJECT_ROOT/dist"
fi
mkdir -p "$OUT_DIR"

STAMP="$(date +%Y%m%d-%H%M%S)"
ARCHIVE="$OUT_DIR/commercial-rag-autodl-${TIER}-${STAMP}.tar.gz"
STAGE="$(mktemp -d /tmp/commercial-rag-pack-XXXXXX)"
STAGE_ROOT="$STAGE/commercial-rag"
mkdir -p "$STAGE_ROOT"

copy_if_exists() {
  local rel="$1"
  if [[ -e "$PROJECT_ROOT/$rel" ]]; then
    local dest_dir
    dest_dir="$(dirname "$STAGE_ROOT/$rel")"
    mkdir -p "$dest_dir"
    cp -a "$PROJECT_ROOT/$rel" "$STAGE_ROOT/$rel"
    echo "[ok]   $rel"
  else
    echo "[skip] 不存在: $rel"
  fi
}

echo ""
echo "=== commercial-rag 打包 tier=$TIER ==="
echo "项目根: $PROJECT_ROOT"
echo "输出:   $ARCHIVE"
echo ""

ALWAYS=(
  README.md
  .gitignore
  requirements.txt
  src
  scripts
  docs
)
for p in "${ALWAYS[@]}"; do copy_if_exists "$p"; done

if [[ "$TIER" =~ ^(minimal|essential|recommended|full)$ ]]; then
  copy_if_exists "data/eval/eval_questions.jsonl"
  shopt -s nullglob
  for f in "$PROJECT_ROOT"/data/eval/eval_*.csv; do
    copy_if_exists "data/eval/$(basename "$f")"
  done
fi

if [[ "$TIER" =~ ^(essential|recommended|full)$ ]]; then
  copy_if_exists "data/parsed/chunks.jsonl"
  copy_if_exists "data/parsed/doc_manifest.jsonl"
  copy_if_exists "data/parsed/documents.jsonl"
  copy_if_exists "data/parsed/chunk_summary.csv"
  copy_if_exists "data/parsed/embed_summary.csv"
  copy_if_exists "data/parsed/parse_summary.csv"
  copy_if_exists "data/vector/milvus.db"
  copy_if_exists "data/vector/bm25_index.pkl"
  copy_if_exists "data/parsed/.gitkeep"
  copy_if_exists "data/vector/.gitkeep"
  copy_if_exists "data/raw_pdfs/.gitkeep"
fi

if [[ "$TIER" =~ ^(recommended|full)$ ]]; then
  copy_if_exists "data/parsed/mineru"
fi

if [[ "$TIER" == "full" ]]; then
  copy_if_exists "data/raw_pdfs"
fi

if [[ "$INCLUDE_EVAL_ARTIFACTS" -eq 1 ]]; then
  copy_if_exists "data/eval/eval_detail.jsonl"
  shopt -s nullglob
  for f in "$PROJECT_ROOT"/data/eval/eval_misses*.jsonl; do
    copy_if_exists "data/eval/$(basename "$f")"
  done
  copy_if_exists "data/eval/eval_rerank_results.csv"
  copy_if_exists "data/eval/eval_rerank_answer_results.csv"
fi

if [[ "$INCLUDE_NOTES" -eq 1 ]]; then
  copy_if_exists "notes"
fi

find "$STAGE_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$STAGE_ROOT" -type d -name ".obsidian" -exec rm -rf {} + 2>/dev/null || true

cat > "$STAGE_ROOT/PACK_MANIFEST.txt" <<EOF
commercial-rag AutoDL pack
tier: $TIER
created: $(date -Iseconds)
project_root: $PROJECT_ROOT

目录说明:
  minimal     代码 + docs + 评测集（服务器全量重跑数据）
  essential   + chunks/milvus/bm25（跳过 embed，适合 POC 迁移）
  recommended + mineru/（跳过 PDF 解析）
  full        + raw_pdfs/（完整 POC）

服务器解压:
  tar -xzf commercial-rag-autodl-*.tar.gz -C /root/autodl-tmp/
  cd /root/autodl-tmp/commercial-rag

Agent 上下文: docs/CURSOR_AGENT_CONTEXT.md
中期实验:     docs/midterm-summary.md
EOF

echo ""
echo "正在压缩..."
tar -czf "$ARCHIVE" -C "$STAGE" commercial-rag
rm -rf "$STAGE"

SIZE_MB="$(du -m "$ARCHIVE" | cut -f1)"
echo ""
echo "完成: $ARCHIVE (${SIZE_MB} MB)"
echo "上传示例: scp \"$ARCHIVE\" root@<autodl-host>:/root/autodl-tmp/"
