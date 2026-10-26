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
setup_log train_sl
ensure_holomotion_root

TRAIN_PY="${TRAIN_PY:-$ROOT_DIR/.venv-holoboxing/bin/python}"
require_file "$TRAIN_PY"

DATA_PREPARED="${DATA_PREPARED:-$ROOT_DIR/data/processed/combined/prepared}"
require_dir "$DATA_PREPARED"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-}"
[[ -n "$PRIOR_CHECKPOINT" ]] || die "未设置 PRIOR_CHECKPOINT"
require_file "$PRIOR_CHECKPOINT"

RESULT_BASE="${RESULT_BASE:-$ROOT_DIR/results}"
OUT_DIR="$(new_output_dir "$RESULT_BASE" "sl")"

echo "[INFO] train_sl"
(cd "$HOLOMOTION_ROOT" && "$TRAIN_PY" -m holoboxing.training.train_sl --data "$DATA_PREPARED" --prior-checkpoint "$PRIOR_CHECKPOINT" --output "$OUT_DIR" --epochs "${SL_EPOCHS:-20}" --batch-size "${SL_BATCH_SIZE:-256}" --device "${DEVICE:-cuda:0}")

echo "[INFO] SL 训练完成: $OUT_DIR"
