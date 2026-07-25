#!/usr/bin/env bash
# 一键启动：Ollama + FastAPI + Vite 前端（支持远程访问）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/data/logs"
PID_DIR="$LOG_DIR/pids"
mkdir -p "$PID_DIR"

# 加载项目根 .env（语义缓存等）
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

API_PORT="${RAG_API_PORT:-8000}"
UI_PORT="${RAG_UI_PORT:-5173}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"

export RAG_COMPARATIVE_ENTITY_RERANK="${RAG_COMPARATIVE_ENTITY_RERANK:-1}"

log() { echo "[start_ui] $*"; }

is_up() {
  local url="$1"
  curl -sf -o /dev/null -m 2 "$url" 2>/dev/null
}

start_bg() {
  local name="$1"
  local pid_file="$PID_DIR/${name}.pid"
  local log_file="$LOG_DIR/${name}.log"
  shift
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    log "$name 已在运行 (pid $(cat "$pid_file"))"
    return 0
  fi
  nohup "$@" >>"$log_file" 2>&1 &
  echo $! >"$pid_file"
  log "$name 已启动 (pid $(cat "$pid_file"))，日志: $log_file"
}

stop_pidfile() {
  local name="$1"
  local pid_file="$PID_DIR/${name}.pid"
  [[ -f "$pid_file" ]] || return 0
  local pid
  pid="$(cat "$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
    log "已停止 $name (pid $pid)"
  fi
  rm -f "$pid_file"
}

# --- Ollama ---
if is_up "http://127.0.0.1:${OLLAMA_PORT}/api/tags"; then
  log "Ollama 已在运行 (:${OLLAMA_PORT})"
else
  start_bg ollama ollama serve
  for _ in $(seq 1 30); do
    is_up "http://127.0.0.1:${OLLAMA_PORT}/api/tags" && break
    sleep 1
  done
  is_up "http://127.0.0.1:${OLLAMA_PORT}/api/tags" || { log "Ollama 启动超时"; exit 1; }
fi

# --- FastAPI ---
if is_up "http://127.0.0.1:${API_PORT}/health"; then
  if [[ "${RAG_RESTART_API:-0}" == "1" ]]; then
    log "RAG_RESTART_API=1，重启 API 以应用环境变量"
    stop_pidfile api 2>/dev/null || true
  else
    log "API 已在运行 (:${API_PORT})"
  fi
fi

if ! is_up "http://127.0.0.1:${API_PORT}/health"; then
  cache_flag="${RAG_SEMANTIC_CACHE_ENABLED:-0}"
  log "启动 API（RAG_SEMANTIC_CACHE_ENABLED=${cache_flag}）"
  start_bg api bash -c "cd '$ROOT' && set -a && [ -f '$ROOT/.env' ] && . '$ROOT/.env'; set +a && uvicorn rag_api:app --host 0.0.0.0 --port ${API_PORT} --app-dir src"
  for _ in $(seq 1 120); do
    is_up "http://127.0.0.1:${API_PORT}/health" && break
    sleep 1
  done
  is_up "http://127.0.0.1:${API_PORT}/health" || { log "API 启动超时（模型加载较慢，请查看 $LOG_DIR/api.log）"; exit 1; }
fi

# --- Vite 前端 ---
VITE_BIN="$ROOT/frontend/node_modules/.bin/vite"
if [[ ! -x "$VITE_BIN" ]]; then
  log "未找到 frontend/node_modules，请先执行: cd frontend && npm install"
  exit 1
fi

if is_up "http://127.0.0.1:${UI_PORT}/"; then
  log "前端已在运行 (:${UI_PORT})"
else
  start_bg vite bash -c "cd '$ROOT/frontend' && '$VITE_BIN' --host 0.0.0.0 --port ${UI_PORT}"
  for _ in $(seq 1 30); do
    is_up "http://127.0.0.1:${UI_PORT}/" && break
    sleep 1
  done
  is_up "http://127.0.0.1:${UI_PORT}/" || { log "前端启动超时，请查看 $LOG_DIR/vite.log"; exit 1; }
fi

# --- 访问地址 ---
LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
PUBLIC_IP="${AUTODL_PUBLIC_IP:-}"

echo ""
echo "=========================================="
echo "  commercial-rag UI 已就绪"
echo "=========================================="
echo "  本机访问:  http://127.0.0.1:${UI_PORT}"
[[ -n "$LAN_IP" ]] && echo "  局域网:    http://${LAN_IP}:${UI_PORT}"
echo "  缓存:      RAG_SEMANTIC_CACHE_ENABLED=${RAG_SEMANTIC_CACHE_ENABLED:-0}"
echo "  重启 API:  RAG_RESTART_API=1 bash scripts/start_ui.sh"
echo ""
echo "  远程访问（任选其一）:"
echo "  1) AutoDL: 控制台 → 自定义服务 → 添加端口 ${UI_PORT} → 用生成的公网链接打开"
echo "  2) SSH 隧道: ssh -L ${UI_PORT}:127.0.0.1:${UI_PORT} user@服务器"
echo "     然后本机浏览器打开 http://localhost:${UI_PORT}"
echo ""
echo "  停止服务:  bash scripts/stop_ui.sh"
echo "  查看日志:  tail -f data/logs/{api,vite,ollama}.log"
echo "=========================================="
