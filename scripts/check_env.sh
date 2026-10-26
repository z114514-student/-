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
setup_log check_env

require_cmd bash
require_cmd python3
ensure_holomotion_root

TRAIN_PY="${TRAIN_PY:-$ROOT_DIR/.venv-holoboxing/bin/python}"
RETARGET_PY="${RETARGET_PY:-$ROOT_DIR/.venv-holoboxing-retarget/bin/python}"

require_file "$TRAIN_PY"
require_file "$RETARGET_PY"

"$TRAIN_PY" -c 'import sys; print("[INFO] train python:", sys.executable)'
"$RETARGET_PY" -c 'import sys; print("[INFO] retarget python:", sys.executable)'

if [[ "${TRAIN_PY}" == "${RETARGET_PY}" ]]; then
  die "训练环境与重定向环境不能共用同一解释器"
fi

if [[ -n "${HOLOBOXING_ASSETS:-}" ]]; then
  require_file "$HOLOBOXING_ASSETS/resources.json"
fi
if [[ -n "${HOLOBOXING_LOWER:-}" ]]; then
  require_file "$HOLOBOXING_LOWER/config.yaml"
  require_file "$HOLOBOXING_LOWER/exported/model_10500.onnx"
fi

echo "[INFO] 环境检查完成"
