#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"
TASK="Unitree-Go2-Velocity-V5-Stairs-Specialist"
EXPERIMENT_NAME="unitree_go2_velocity_v5_stairs_specialist"
FOUNDATION_CHECKPOINT="${FOUNDATION_CHECKPOINT:-${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity_v5_foundation/2026-08-26_23-13-40_multiterrain_v5_perceptive_gru_foundation_balanced4_fromscratch_seed42_env1024_i5000/model_3200.pt}"
EXPECTED_FOUNDATION_SHA256="c23c592eb9401c0173e470672cfb67aa1bf12a2074cb8fbe93a438f2c730ed08"
CONTINUE_CHECKPOINT="${CONTINUE_CHECKPOINT:-}"
EXPECTED_CONTINUE_SHA256="${EXPECTED_CONTINUE_SHA256:-}"

NUM_ENVS="${NUM_ENVS:-1024}"
SEED="${SEED:-42}"
MAX_ITERATIONS="${MAX_ITERATIONS:-4000}"
RUN_STAGE="${RUN_STAGE:-stairs_specialist_transfer}"
RUN_NAME="${V5_STAIRS_RUN_NAME:-v5_perceptive_gru_stairs_specialist_${RUN_STAGE}_from_foundation3200_trainseed${SEED}_env${NUM_ENVS}_i${MAX_ITERATIONS}}"
LOGGER="${LOGGER:-tensorboard}"
DEVICE="${DEVICE:-cuda:0}"
ENABLE_GPU_MONITOR="${ENABLE_GPU_MONITOR:-1}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-5}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -n "$CONTINUE_CHECKPOINT" ]]; then
    resume_checkpoint="$CONTINUE_CHECKPOINT"
    expected_resume_sha256="$EXPECTED_CONTINUE_SHA256"
    resume_mode="continue specialist with optimizer and iteration state"
    reset_optimizer=0
    reset_iteration=0
else
    resume_checkpoint="$FOUNDATION_CHECKPOINT"
    expected_resume_sha256="$EXPECTED_FOUNDATION_SHA256"
    resume_mode="transfer formal foundation weights"
    reset_optimizer=1
    reset_iteration=1
fi

timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
console_log="${V5_STAIRS_CONSOLE_LOG:-${WORKSPACE_DIR}/unitree_go2_${RUN_NAME}_${timestamp}.log}"
gpu_log="${V5_STAIRS_GPU_LOG:-${WORKSPACE_DIR}/gpu_${RUN_NAME}_${timestamp}.csv}"
log_root="${PROJECT_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}"
gpu_monitor_pid=""

usage() {
    cat <<'EOF'
Train or continue the Unitree Go2 v5 stairs specialist.

By default, policy and observation-normalization weights are transferred from
the formal foundation model_3200 while optimizer and iteration start fresh.
Set CONTINUE_CHECKPOINT and EXPECTED_CONTINUE_SHA256 to continue a specialist
checkpoint with its optimizer and iteration state. Existing artifacts are
never overwritten.

Defaults:
  NUM_ENVS=1024
  SEED=42
  MAX_ITERATIONS=4000
  RUN_STAGE=stairs_specialist_transfer
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
if [[ -n "$CONTINUE_CHECKPOINT" && -z "$EXPECTED_CONTINUE_SHA256" ]]; then
    echo "EXPECTED_CONTINUE_SHA256 is required with CONTINUE_CHECKPOINT" >&2
    exit 2
fi
if [[ ! -f "$resume_checkpoint" ]]; then
    echo "Resume checkpoint not found: $resume_checkpoint" >&2
    exit 1
fi
resume_sha256="$(sha256sum "$resume_checkpoint" | awk '{print $1}')"
if [[ "$resume_sha256" != "$expected_resume_sha256" ]]; then
    echo "Resume checkpoint SHA-256 mismatch." >&2
    echo "Expected: $expected_resume_sha256" >&2
    echo "Actual:   $resume_sha256" >&2
    exit 1
fi
if find "$log_root" -mindepth 1 -maxdepth 1 -type d -name "*_${RUN_NAME}" -print -quit 2>/dev/null | grep -q .; then
    echo "Refusing to reuse an existing v5 stairs run name: $RUN_NAME" >&2
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

echo "Task:              $TASK"
echo "Task variant:      v5_stairs_specialist"
echo "Architecture:      recurrent perceptive actor / asymmetric critic"
echo "Resume mode:       $resume_mode"
echo "Resume checkpoint: $resume_checkpoint"
echo "Resume SHA256:     $resume_sha256"
if [[ "$reset_optimizer" == "1" ]]; then
    echo "Optimizer state:   reset (not loaded)"
    echo "Iteration counter: reset to 0"
else
    echo "Optimizer state:   loaded from specialist checkpoint"
    echo "Iteration counter: continued from specialist checkpoint"
fi
echo "Terrains:          flat/rough rehearsal + ascending/descending stairs"
echo "Experiment:        $EXPERIMENT_NAME"
echo "Environments:      $NUM_ENVS"
echo "Training seed:     $SEED"
echo "Eval scenario seed: N/A (training command)"
echo "Iterations:        $MAX_ITERATIONS"
echo "Run stage:         $RUN_STAGE"
echo "Run name:          $RUN_NAME"
echo "Device:            $DEVICE"
echo "Git HEAD:          $(git rev-parse HEAD)"
echo "Console log:       $console_log"
if [[ "$ENABLE_GPU_MONITOR" == "1" ]]; then
    echo "GPU log:           $gpu_log"
else
    echo "GPU log:           disabled"
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
    --resume
    --resume_path "$resume_checkpoint"
)

if [[ "$reset_optimizer" == "1" ]]; then
    train_command+=(--reset_optimizer_on_resume)
fi
if [[ "$reset_iteration" == "1" ]]; then
    train_command+=(--reset_iteration_on_resume)
fi

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
    --why="Unitree Go2 v5 stairs specialist training ${RUN_NAME}" \
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
