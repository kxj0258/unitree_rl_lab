#!/usr/bin/env bash
set -Eeuo pipefail
cd "${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
OUTPUT="${OUTPUT:-$PWD/evaluation/proxy_teacher_dispatcher_online_smoke_seed20260829_steps64_v1.json}"
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite: $OUTPUT" >&2; exit 1; }
CONSOLE_LOG="${CONSOLE_LOG:-${OUTPUT%.json}.console.log}"
[[ ! -e "$CONSOLE_LOG" ]] || { echo "Refusing to overwrite: $CONSOLE_LOG" >&2; exit 1; }
set +e
/home/kxj/miniforge3/envs/unitree_rl/bin/python scripts/project/smoke_go2_proxy_teacher_dispatcher_online.py \
  --registry "$PWD/config/proxy_teacher_registry_v1.json" --output "$OUTPUT" \
  --steps "${STEPS:-64}" --seed "${SCENARIO_SEED:-20260829}" --device "${DEVICE:-cuda:0}" \
  2>&1 | tee "$CONSOLE_LOG"
status=${PIPESTATUS[0]}
set -e
echo "Online smoke exit code: $status"
echo "Console log: $CONSOLE_LOG"
if [[ $status -eq 0 ]]; then
  echo "Output JSON: $OUTPUT"
fi
exit "$status"
