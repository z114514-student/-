#!/usr/bin/env bash
set -euo pipefail

now_ts() { date +"%Y%m%d-%H%M%S"; }

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_file() {
  local f="$1"
  [[ -f "$f" ]] || die "缺少文件: $f"
}

require_dir() {
  local d="$1"
  [[ -d "$d" ]] || die "缺少目录: $d"
}

require_cmd() {
  local c="$1"
  command -v "$c" >/dev/null 2>&1 || die "缺少命令: $c"
}

load_config() {
  local cfg="${1:-}"
  [[ -n "$cfg" ]] || return 0
  require_file "$cfg"
  # shellcheck disable=SC1090
  source "$cfg"
}

setup_log() {
  local stage="$1"
  local log_root="${LOG_ROOT:-./logs}"
  mkdir -p "$log_root"
  local log_file="$log_root/${stage}-$(now_ts).log"
  exec > >(tee -a "$log_file") 2>&1
  echo "[INFO] 日志: $log_file"
}

new_output_dir() {
  local base="$1"
  local prefix="$2"
  mkdir -p "$base"
  local d="$base/${prefix}-$(now_ts)"
  mkdir -p "$d"
  echo "$d"
}

ensure_holomotion_root() {
  [[ -n "${HOLOMOTION_ROOT:-}" ]] || die "未设置 HOLOMOTION_ROOT"
  require_dir "$HOLOMOTION_ROOT"
  require_file "$HOLOMOTION_ROOT/holoboxing/run.sh"
}
