#!/usr/bin/env bash
# 停止 start_ui.sh 启动的 Ollama / API / Vite
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_DIR="$ROOT/data/logs/pids"

log() { echo "[stop_ui] $*"; }

stop_pidfile() {
  local name="$1"
  local pid_file="$PID_DIR/${name}.pid"
  if [[ ! -f "$pid_file" ]]; then
    return 0
  fi
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

stop_pidfile vite
stop_pidfile api
stop_pidfile ollama

log "完成"
