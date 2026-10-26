#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/common.sh"

CONFIG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    *) die "未知参数: $1" ;;
  esac
done

load_config "$CONFIG"
setup_log data_prepare
ensure_holomotion_root

TRAIN_PY="${TRAIN_PY:-$ROOT_DIR/.venv-holoboxing/bin/python}"
RETARGET_PY="${RETARGET_PY:-$ROOT_DIR/.venv-holoboxing-retarget/bin/python}"
require_file "$TRAIN_PY"
require_file "$RETARGET_PY"

DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
ROBO_OUT="${ROBO_OUT:-$DATA_ROOT/raw/RoboStriker}"
INTERHUMAN_OUT="${INTERHUMAN_OUT:-$DATA_ROOT/raw/InterHuman}"
RETARGET_OUT="${RETARGET_OUT:-$DATA_ROOT/converted/InterHuman}"
PREPARED_OUT="${PREPARED_OUT:-$DATA_ROOT/processed/combined}"
WORKERS="${WORKERS:-1}"
SMPL_MODEL="${SMPL_MODEL:-}"
require_dir "$(dirname "$ROBO_OUT")"
mkdir -p "$ROBO_OUT" "$INTERHUMAN_OUT" "$RETARGET_OUT" "$PREPARED_OUT"

[[ -n "${HOLOBOXING_ASSETS:-}" ]] || die "未设置 HOLOBOXING_ASSETS"
require_file "$HOLOBOXING_ASSETS/resources.json"

echo "[INFO] 下载 RoboStriker"
(cd "$HOLOMOTION_ROOT" && "$TRAIN_PY" -m holoboxing.data download robostriker --output "$ROBO_OUT" --workers "$WORKERS")

echo "[INFO] 下载 InterHuman"
(cd "$HOLOMOTION_ROOT" && "$TRAIN_PY" -m holoboxing.data download interhuman --output "$INTERHUMAN_OUT" --workers "$WORKERS")

[[ -n "$SMPL_MODEL" ]] || die "未设置 SMPL_MODEL（受许可资源，需用户自行准备）"
require_file "$SMPL_MODEL"

echo "[INFO] 重定向 InterHuman"
(cd "$HOLOMOTION_ROOT" && "$RETARGET_PY" -m holoboxing.data retarget --source "$INTERHUMAN_OUT" --output "$RETARGET_OUT" --smpl-model "$SMPL_MODEL" --assets "$HOLOBOXING_ASSETS")

echo "[INFO] prepare 数据"
(cd "$HOLOMOTION_ROOT" && "$TRAIN_PY" -m holoboxing.data prepare --source "$ROBO_OUT" --source "$RETARGET_OUT" --output "$PREPARED_OUT" --workers "$WORKERS" --assets "$HOLOBOXING_ASSETS")

echo "[INFO] verify 数据"
(cd "$HOLOMOTION_ROOT" && "$TRAIN_PY" -m holoboxing.data verify --output "$PREPARED_OUT" --assets "$HOLOBOXING_ASSETS")

echo "[INFO] 数据准备与验证完成"
