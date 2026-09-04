#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"
TASK="Unitree-Go2-Velocity-V5-Foundation"
EXPERIMENT_NAME="unitree_go2_velocity_v5_foundation"

NUM_ENVS="${NUM_ENVS:-1024}"
SEED="${SEED:-42}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
RUN_STAGE="${RUN_STAGE:-foundation_from_scratch}"
RUN_NAME="${V5_RUN_NAME:-v5_perceptive_gru_foundation_balanced4_${RUN_STAGE}_seed${SEED}_env${NUM_ENVS}_i${MAX_ITERATIONS}}"
LOGGER="${LOGGER:-tensorboard}"
DEVICE="${DEVICE:-cuda:0}"
ENABLE_GPU_MONITOR="${ENABLE_GPU_MONITOR:-1}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-5}"
DRY_RUN="${DRY_RUN:-0}"

timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
console_log="${V5_CONSOLE_LOG:-${WORKSPACE_DIR}/unitree_go2_${RUN_NAME}_${timestamp}.log}"
gpu_log="${V5_GPU_LOG:-${WORKSPACE_DIR}/gpu_${RUN_NAME}_${timestamp}.csv}"
log_root="${PROJECT_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}"
gpu_monitor_pid=""

usage() {
    cat <<'EOF'
Train the Unitree Go2 v5 perceptive recurrent foundation policy from scratch.

This stage never accepts a resume checkpoint.  It trains on balanced flat,
random rough, ascending slope, and descending slope terrains.  Existing run,
console, and GPU-log paths are never overwritten.

Defaults:
  NUM_ENVS=1024
  SEED=42
  MAX_ITERATIONS=5000
  RUN_STAGE=foundation_from_scratch
  DEVICE=cuda:0
  LOGGER=tensorboard
  ENABLE_GPU_MONITOR=1
  GPU_POLL_SECONDS=5
  DRY_RUN=0
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
for name in NUM_ENVS MAX_ITERATIONS GPU_POLL_SECONDS; do
    value="${!name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer: $value" >&2
        exit 2
    fi
done
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "SEED must be a non-negative integer: $SEED" >&2
    exit 2
fi
if [[ "$ENABLE_GPU_MONITOR" != "0" && "$ENABLE_GPU_MONITOR" != "1" ]]; then
    echo "ENABLE_GPU_MONITOR must be 0 or 1: $ENABLE_GPU_MONITOR" >&2
    exit 2
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1: $DRY_RUN" >&2
    exit 2
fi
if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "Project directory not found: $PROJECT_DIR" >&2
    exit 1
fi
if find "$log_root" -mindepth 1 -maxdepth 1 -type d -name "*_${RUN_NAME}" -print -quit 2>/dev/null | grep -q .; then
    echo "Refusing to reuse an existing v5 run name: $RUN_NAME" >&2
    exit 1
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
echo "Task variant:   v5_foundation"
echo "Architecture:   recurrent perceptive actor / asymmetric critic"
echo "Initialization: random (resume disabled)"
echo "Terrains:       flat, random_rough, pyramid_slope, inverted_pyramid_slope"
echo "Experiment:     $EXPERIMENT_NAME"
echo "Environments:   $NUM_ENVS"
echo "Training seed:  $SEED"
echo "Iterations:     $MAX_ITERATIONS"
echo "Run stage:      $RUN_STAGE"
echo "Run name:       $RUN_NAME"
echo "Device:         $DEVICE"
echo "Git HEAD:       $(git rev-parse HEAD)"
echo "Console log:    $console_log"
if [[ "$ENABLE_GPU_MONITOR" == "1" ]]; then
    echo "GPU log:        $gpu_log"
else
    echo "GPU log:        disabled"
fi

train_command=(
    ./unitree_rl_lab.sh --train
    --task "$TASK"
    --num_envs "$NUM_ENVS"
    --seed "$SEED"
    --max_iterations "$MAX_ITERATIONS"
    --run_name "$RUN_NAME"
    --experiment_name "$EXPERIMENT_NAME"
    --logger "$LOGGER"
    --device "$DEVICE"
)

if [[ "$DRY_RUN" == "1" ]]; then
    printf 'Dry-run command:'
    printf ' %q' "${train_command[@]}"
    printf '\n'
    exit 0
fi

if [[ "$ENABLE_GPU_MONITOR" == "1" ]]; then
    gpu_query_fields="timestamp,index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
    nvidia-smi --query-gpu="$gpu_query_fields" --format=csv --loop="$GPU_POLL_SECONDS" >"$gpu_log" 2>&1 &
    gpu_monitor_pid=$!
    echo "GPU monitor PID: $gpu_monitor_pid"
fi

set +e
systemd-inhibit \
    --what=sleep \
    --why="Unitree Go2 v5 foundation training ${RUN_NAME}" \
    "${train_command[@]}" \
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
