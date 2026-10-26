#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/common.sh"

MODE="${1:-}"
[[ -n "$MODE" ]] || die "用法: $0 <validate|benchmark|match|view> --config <file> [--opponent path] [--replay path]"
shift

CONFIG=""
OPPONENT=""
REPLAY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --opponent) OPPONENT="$2"; shift 2 ;;
    --replay) REPLAY="$2"; shift 2 ;;
    *) die "未知参数: $1" ;;
  esac
done

load_config "$CONFIG"
setup_log "eval_${MODE}"
ensure_holomotion_root

MODEL_ONNX="${MODEL_ONNX:-}"
[[ -n "$MODEL_ONNX" ]] || die "未设置 MODEL_ONNX"
require_file "$MODEL_ONNX"
RUN_SH="$HOLOMOTION_ROOT/holoboxing/run.sh"
require_file "$RUN_SH"

RESULT_BASE="${RESULT_BASE:-$ROOT_DIR/results}"
OUT_DIR="$(new_output_dir "$RESULT_BASE" "eval-${MODE}")"

case "$MODE" in
  validate)
    "$RUN_SH" --validate "$MODEL_ONNX" | tee "$OUT_DIR/validate.txt"
    ;;
  benchmark)
    "$RUN_SH" --validate "$MODEL_ONNX" --benchmark --output "$OUT_DIR/benchmark.json"
    ;;
  match)
    [[ -n "$OPPONENT" ]] || die "match 模式必须指定 --opponent"
    require_file "$OPPONENT"
    "$RUN_SH" --match "$MODEL_ONNX" "$OPPONENT" --output "$OUT_DIR/match"
    ;;
  view)
    [[ -n "$REPLAY" ]] || die "view 模式必须指定 --replay"
    require_file "$REPLAY"
    "$RUN_SH" --view "$REPLAY"
    ;;
  *)
    die "不支持的模式: $MODE"
    ;;
esac

echo "[INFO] 完成: $MODE"
