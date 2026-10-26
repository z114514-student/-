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
setup_log export_onnx
ensure_holomotion_root

TRAIN_PY="${TRAIN_PY:-$ROOT_DIR/.venv-holoboxing/bin/python}"
require_file "$TRAIN_PY"
CHECKPOINT="${CHECKPOINT:-}"
[[ -n "$CHECKPOINT" ]] || die "未设置 CHECKPOINT"
require_file "$CHECKPOINT"

RESULT_BASE="${RESULT_BASE:-$ROOT_DIR/results}"
OUT_DIR="$(new_output_dir "$RESULT_BASE" "onnx")"
ONNX_OUT="${ONNX_OUT:-$OUT_DIR/model.onnx}"

echo "[INFO] export onnx"
(cd "$HOLOMOTION_ROOT" && "$TRAIN_PY" -m holoboxing.training.export --checkpoint "$CHECKPOINT" --output "$ONNX_OUT")

echo "[INFO] 导出完成: $ONNX_OUT"
