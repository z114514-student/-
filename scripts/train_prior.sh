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
setup_log train_prior
ensure_holomotion_root

TRAIN_PY="${TRAIN_PY:-$ROOT_DIR/.venv-holoboxing/bin/python}"
require_file "$TRAIN_PY"

ROBO_OUT="${ROBO_OUT:-$ROOT_DIR/data/raw/RoboStriker}"
RETARGET_OUT="${RETARGET_OUT:-$ROOT_DIR/data/converted/InterHuman}"
[[ -n "${HOLOBOXING_ASSETS:-}" ]] || die "未设置 HOLOBOXING_ASSETS"
require_file "$HOLOBOXING_ASSETS/resources.json"

RESULT_BASE="${RESULT_BASE:-$ROOT_DIR/results}"
OUT_DIR="$(new_output_dir "$RESULT_BASE" "prior")"
PRIOR_DATA="${PRIOR_DATA:-$OUT_DIR/prior-data}"
mkdir -p "$PRIOR_DATA"

echo "[INFO] prepare_prior"
(cd "$HOLOMOTION_ROOT" && "$TRAIN_PY" -m holoboxing.training.prepare_prior --source "$ROBO_OUT" --source "$RETARGET_OUT" --assets "$HOLOBOXING_ASSETS" --output "$PRIOR_DATA")

echo "[INFO] train_prior"
(cd "$HOLOMOTION_ROOT" && "$TRAIN_PY" -m holoboxing.training.train_prior --data "$PRIOR_DATA" --output "$OUT_DIR" --epochs "${PRIOR_EPOCHS:-20}" --batch-size "${PRIOR_BATCH_SIZE:-256}" --device "${DEVICE:-cuda:0}")

echo "[INFO] prior 训练完成: $OUT_DIR"
