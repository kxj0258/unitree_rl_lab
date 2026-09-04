#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"

TASK_VARIANT="${TASK_VARIANT:-survival}"
EXPERIMENT_NAME="unitree_go2_velocity_multiterrain"
case "$TASK_VARIANT" in
    survival) TASK="Unitree-Go2-Velocity-MultiTerrain" ;;
    tracking_aware) TASK="Unitree-Go2-Velocity-MultiTerrain-TrackingAware" ;;
    height_scan) TASK="Unitree-Go2-Velocity-MultiTerrain-HeightScan" ;;
    *)
        echo "TASK_VARIANT must be survival, tracking_aware, or height_scan: $TASK_VARIANT" >&2
        exit 2
        ;;
esac
NUM_ENVS="${NUM_ENVS:-2048}"
SEED="${SEED:-42}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2000}"
RUN_STAGE="${RUN_STAGE:-prototype}"
RUN_NAME="${MULTITERRAIN_RUN_NAME:-multiterrain_v1_challenging_noheightscan_${RUN_STAGE}_seed${SEED}_env${NUM_ENVS}_i${MAX_ITERATIONS}}"
LOGGER="${LOGGER:-tensorboard}"
DEVICE="${DEVICE:-cuda:0}"
ENABLE_GPU_MONITOR="${ENABLE_GPU_MONITOR:-1}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-5}"
RESUME_RUN="${RESUME_RUN:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
gpu_query_fields="timestamp,index,name,driver_version,temperature.gpu,"
gpu_query_fields+="utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"

timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
console_log="${MULTITERRAIN_CONSOLE_LOG:-${WORKSPACE_DIR}/unitree_go2_${RUN_NAME}_${timestamp}.log}"
gpu_log="${MULTITERRAIN_GPU_LOG:-${WORKSPACE_DIR}/gpu_${RUN_NAME}_${timestamp}.csv}"
gpu_monitor_pid=""

usage() {
    cat <<'EOF'
Train the registered Unitree-Go2-Velocity-MultiTerrain task.

The task is fixed for safety. Environment variables may override the run budget
and output naming, and existing logs are never overwritten.

Optional output overrides use the task-specific variables
MULTITERRAIN_RUN_NAME, MULTITERRAIN_CONSOLE_LOG, and MULTITERRAIN_GPU_LOG.

Defaults:
  NUM_ENVS=2048
  SEED=42
  MAX_ITERATIONS=2000
  RUN_STAGE=prototype
  LOGGER=tensorboard
  DEVICE=cuda:0
  ENABLE_GPU_MONITOR=1
  GPU_POLL_SECONDS=5
  RESUME_RUN=                 # existing run directory name; requires RESUME_CHECKPOINT
  RESUME_CHECKPOINT=          # checkpoint filename; requires RESUME_RUN
  TASK_VARIANT=survival       # survival, tracking_aware, or height_scan

Examples:
  RUN_STAGE=smoke NUM_ENVS=64 MAX_ITERATIONS=5 scripts/project/train_go2_multiterrain.sh
  RUN_STAGE=pilot NUM_ENVS=1024 MAX_ITERATIONS=500 scripts/project/train_go2_multiterrain.sh
  RUN_STAGE=prototype NUM_ENVS=2048 MAX_ITERATIONS=2000 scripts/project/train_go2_multiterrain.sh
  RUN_STAGE=prototype_resume MAX_ITERATIONS=1500 \
    RESUME_RUN=2026-..._pilot... RESUME_CHECKPOINT=model_499.pt \
    scripts/project/train_go2_multiterrain.sh
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
if [[ "$ENABLE_GPU_MONITOR" != "0" && "$ENABLE_GPU_MONITOR" != "1" ]]; then
    echo "ENABLE_GPU_MONITOR must be 0 or 1: $ENABLE_GPU_MONITOR" >&2
    exit 2
fi
if [[ -n "$RESUME_RUN" || -n "$RESUME_CHECKPOINT" ]]; then
    if [[ -z "$RESUME_RUN" || -z "$RESUME_CHECKPOINT" ]]; then
        echo "RESUME_RUN and RESUME_CHECKPOINT must be provided together" >&2
        exit 2
    fi
    resume_path="${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity_multiterrain/${RESUME_RUN}/${RESUME_CHECKPOINT}"
    if [[ ! -f "$resume_path" ]]; then
        echo "Resume checkpoint not found: $resume_path" >&2
        exit 1
    fi
    # Isaac Lab resolves --checkpoint as a regular expression. Anchor and
    # escape dots so checksum/metadata sidecars cannot be selected instead.
    resume_checkpoint_regex="^${RESUME_CHECKPOINT//./\\.}$"
    resume_args=(--resume --load_run "^${RESUME_RUN}$" --checkpoint "$resume_checkpoint_regex")
else
    resume_args=()
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
echo "Task variant:   $TASK_VARIANT"
echo "Experiment:     $EXPERIMENT_NAME"
echo "Environments:   $NUM_ENVS"
echo "Training seed:  $SEED"
echo "Iterations:     $MAX_ITERATIONS"
echo "Run stage:      $RUN_STAGE"
echo "Run name:       $RUN_NAME"
echo "Device:         $DEVICE"
echo "Git HEAD:       $(git rev-parse HEAD)"
if [[ ${#resume_args[@]} -gt 0 ]]; then
    echo "Resume run:     $RESUME_RUN"
    echo "Resume ckpt:    $RESUME_CHECKPOINT"
else
    echo "Resume:         disabled"
fi
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
    --why="Unitree Go2 multiterrain training ${RUN_NAME}" \
    ./unitree_rl_lab.sh --train \
    --task "$TASK" \
    --num_envs "$NUM_ENVS" \
    --seed "$SEED" \
    --max_iterations "$MAX_ITERATIONS" \
    --run_name "$RUN_NAME" \
    --experiment_name "$EXPERIMENT_NAME" \
    --logger "$LOGGER" \
    --device "$DEVICE" \
    "${resume_args[@]}" \
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
