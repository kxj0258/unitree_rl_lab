#!/usr/bin/env bash
set -Eeuo pipefail
cd "${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"

OUTPUT_DIR="${OUTPUT_DIR:-$PWD/logs/student_smoke/proxy_teacher_imitation_trainseed42_steps128_v1}"
CONSOLE_LOG="${CONSOLE_LOG:-${OUTPUT_DIR}.console.log}"
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Refusing to overwrite output directory: $OUTPUT_DIR" >&2; exit 1; }
[[ ! -e "$CONSOLE_LOG" ]] || { echo "Refusing to overwrite console log: $CONSOLE_LOG" >&2; exit 1; }
mkdir -p "$(dirname "$CONSOLE_LOG")"

set +e
/home/kxj/miniforge3/envs/unitree_rl/bin/python scripts/project/smoke_go2_proxy_teacher_student_imitation.py \
  --registry "$PWD/config/proxy_teacher_registry_v1.json" \
  --output-dir "$OUTPUT_DIR" \
  --steps "${STEPS:-128}" \
  --training-seed "${TRAINING_SEED:-42}" \
  --learning-rate "${LEARNING_RATE:-0.0003}" \
  --max-grad-norm "${MAX_GRAD_NORM:-1.0}" \
  --device "${DEVICE:-cuda:0}" \
  2>&1 | tee "$CONSOLE_LOG"
status=${PIPESTATUS[0]}
set -e

result_json="$OUTPUT_DIR/result.json"
checkpoint="$OUTPUT_DIR/student_model_${STEPS:-128}.pt"
if [[ $status -eq 0 && ( ! -s "$result_json" || ! -s "$checkpoint" ) ]]; then
  echo "ERROR: Python returned zero but required result artifacts are missing or empty." >&2
  status=1
fi
if [[ $status -eq 0 ]] && ! /home/kxj/miniforge3/envs/unitree_rl/bin/python -c \
  'import json,sys; result=json.load(open(sys.argv[1])); raise SystemExit(0 if result.get("status") == "pass" else 1)' \
  "$result_json"; then
  echo "ERROR: result.json does not report status=pass." >&2
  status=1
fi

echo "Student imitation smoke exit code: $status"
echo "Console log: $CONSOLE_LOG"
if [[ $status -eq 0 ]]; then
  echo "Output directory: $OUTPUT_DIR"
  echo "Result JSON: $result_json"
  echo "Checkpoint: $checkpoint"
fi
exit "$status"
