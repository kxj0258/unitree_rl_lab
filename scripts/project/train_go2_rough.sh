#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"

TASK="${TASK:-Unitree-Go2-Velocity-Rough}"
NUM_ENVS="${NUM_ENVS:-2048}"
SEED="${SEED:-42}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3000}"
RUN_STAGE="${RUN_STAGE:-formal}"
RUN_NAME="${RUN_NAME:-roughmix50_curriculum_${RUN_STAGE}_seed${SEED}_env${NUM_ENVS}_i${MAX_ITERATIONS}}"
LOGGER="${LOGGER:-tensorboard}"
DEVICE="${DEVICE:-cuda:0}"
ENABLE_GPU_MONITOR="${ENABLE_GPU_MONITOR:-1}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-5}"
gpu_query_fields="timestamp,index,name,driver_version,temperature.gpu,"
gpu_query_fields+="utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"

timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
console_log="${CONSOLE_LOG:-${WORKSPACE_DIR}/unitree_go2_${RUN_NAME}_${timestamp}.log}"
gpu_log="${GPU_LOG:-${WORKSPACE_DIR}/gpu_${RUN_NAME}_${timestamp}.csv}"
gpu_monitor_pid=""

usage() {
    cat <<'EOF'
Train the registered Unitree-Go2-Velocity-Rough task.

The script uses environment variables for overrides and refuses to overwrite logs.

Defaults:
  TASK=Unitree-Go2-Velocity-Rough
  NUM_ENVS=2048
  SEED=42
  MAX_ITERATIONS=3000
  RUN_STAGE=formal
  LOGGER=tensorboard
  DEVICE=cuda:0
  ENABLE_GPU_MONITOR=1
  GPU_POLL_SECONDS=5

Examples:
  scripts/project/train_go2_rough.sh
  RUN_STAGE=pilot MAX_ITERATIONS=500 scripts/project/train_go2_rough.sh
  RUN_STAGE=scalecheck MAX_ITERATIONS=100 scripts/project/train_go2_rough.sh
EOF
}

cleanup() {
    if [[ -n "$gpu_monitor_pid" ]] && kill -0 "$gpu_monitor_pid" 2>/dev/null; then
        kill "$gpu_monitor_pid" 2>/dev/null || true
        wait "$gpu_monitor_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
if [[ $# -ne 0 ]]; then
    usage >&2
    exit 2
fi
if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "Project directory not found: $PROJECT_DIR" >&2
    exit 1
fi
if [[ "$TASK" != "Unitree-Go2-Velocity-Rough" ]]; then
    echo "Refusing unexpected task for rough training: $TASK" >&2
    exit 2
fi
if [[ ! "$NUM_ENVS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_ENVS must be a positive integer: $NUM_ENVS" >&2
    exit 2
fi
if [[ ! "$MAX_ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_ITERATIONS must be a positive integer: $MAX_ITERATIONS" >&2
    exit 2
fi
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "SEED must be a non-negative integer: $SEED" >&2
    exit 2
fi
if [[ ! "$GPU_POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU_POLL_SECONDS must be a positive integer: $GPU_POLL_SECONDS" >&2
    exit 2
fi
if [[ -e "$console_log" ]]; then
    echo "Refusing to overwrite existing console log: $console_log" >&2
    exit 1
fi
if [[ "$ENABLE_GPU_MONITOR" == "1" && -e "$gpu_log" ]]; then
    echo "Refusing to overwrite existing GPU log: $gpu_log" >&2
    exit 1
fi
if [[ "$ENABLE_GPU_MONITOR" == "1" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required when ENABLE_GPU_MONITOR=1" >&2
    exit 1
fi

cd "$PROJECT_DIR"

echo "Task:           $TASK"
echo "Environments:   $NUM_ENVS"
echo "Seed:           $SEED"
echo "Iterations:     $MAX_ITERATIONS"
echo "Run stage:      $RUN_STAGE"
echo "Run name:       $RUN_NAME"
echo "Device:         $DEVICE"
echo "Console log:    $console_log"
if [[ "$ENABLE_GPU_MONITOR" == "1" ]]; then
    echo "GPU log:        $gpu_log"
else
    echo "GPU log:        disabled"
fi

if [[ "$ENABLE_GPU_MONITOR" == "1" ]]; then
    nvidia-smi \
        --query-gpu="$gpu_query_fields" \
        --format=csv \
        --loop="$GPU_POLL_SECONDS" \
        >"$gpu_log" 2>&1 &
    gpu_monitor_pid=$!
    echo "GPU monitor PID: $gpu_monitor_pid"
fi

set +e
systemd-inhibit \
    --what=sleep \
    --why="Unitree Go2 rough training ${RUN_NAME}" \
    ./unitree_rl_lab.sh --train \
    --task "$TASK" \
    --num_envs "$NUM_ENVS" \
    --seed "$SEED" \
    --max_iterations "$MAX_ITERATIONS" \
    --run_name "$RUN_NAME" \
    --logger "$LOGGER" \
    --device "$DEVICE" \
    2>&1 | tee "$console_log"
training_status=${PIPESTATUS[0]}
set -e

cleanup
gpu_monitor_pid=""

echo "Training exit code: $training_status"
echo "Console log: $console_log"
if [[ "$ENABLE_GPU_MONITOR" == "1" ]]; then
    echo "GPU log: $gpu_log"
fi
exit "$training_status"
