#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"
TASK="Unitree-Go2-Velocity-V5-Stairs-Ascending-Remediation"
EXPERIMENT_NAME="unitree_go2_velocity_v5_stairs_ascending_remediation"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity_v5_stairs_specialist/2026-08-27_16-17-34_v5_perceptive_gru_stairs_specialist_formal_continue_from2200_trainseed42_env1024_addi2800_total5000/model_4999.pt}"
EXPECTED_SOURCE_SHA256="47b42309c03450c88f27ceaf66ed393abfcc5a488e7a508a369055ed4cfcee50"

NUM_ENVS="${NUM_ENVS:-128}"
SEED="${SEED:-42}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5}"
RUN_STAGE="${RUN_STAGE:-smoke}"
RUN_NAME="${V5_ASCENDING_RUN_NAME:-v5_stairs_ascending_remediation_${RUN_STAGE}_from4999_trainseed${SEED}_env${NUM_ENVS}_i${MAX_ITERATIONS}}"
LOGGER="${LOGGER:-tensorboard}"
DEVICE="${DEVICE:-cuda:0}"
ENABLE_GPU_MONITOR="${ENABLE_GPU_MONITOR:-1}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-5}"
DRY_RUN="${DRY_RUN:-0}"

timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
console_log="${V5_ASCENDING_CONSOLE_LOG:-${WORKSPACE_DIR}/unitree_go2_${RUN_NAME}_${timestamp}.log}"
gpu_log="${V5_ASCENDING_GPU_LOG:-${WORKSPACE_DIR}/gpu_${RUN_NAME}_${timestamp}.csv}"
log_root="${PROJECT_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}"
gpu_monitor_pid=""

cleanup() {
    if [[ -n "$gpu_monitor_pid" ]] && kill -0 "$gpu_monitor_pid" 2>/dev/null; then
        kill "$gpu_monitor_pid" 2>/dev/null || true
        wait "$gpu_monitor_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

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
if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
    echo "Source checkpoint not found: $SOURCE_CHECKPOINT" >&2
    exit 1
fi
source_sha256="$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')"
if [[ "$source_sha256" != "$EXPECTED_SOURCE_SHA256" ]]; then
    echo "Source checkpoint SHA-256 mismatch." >&2
    echo "Expected: $EXPECTED_SOURCE_SHA256" >&2
    echo "Actual:   $source_sha256" >&2
    exit 1
fi
if find "$log_root" -mindepth 1 -maxdepth 1 -type d -name "*_${RUN_NAME}" -print -quit 2>/dev/null | grep -q .; then
    echo "Refusing to reuse an existing remediation run name: $RUN_NAME" >&2
    exit 1
fi
if [[ -e "$console_log" || ( "$ENABLE_GPU_MONITOR" == "1" && -e "$gpu_log" ) ]]; then
    echo "Refusing to overwrite an existing console or GPU log." >&2
    exit 1
fi
if [[ "$ENABLE_GPU_MONITOR" == "1" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required when ENABLE_GPU_MONITOR=1" >&2
    exit 1
fi

cd "$PROJECT_DIR"
echo "Task:               $TASK"
echo "Task variant:       v5_stairs_ascending_remediation"
echo "Source checkpoint:  $SOURCE_CHECKPOINT"
echo "Source SHA256:      $source_sha256"
echo "Training seed:      $SEED"
echo "Eval scenario seed: N/A (training command)"
echo "Environments:       $NUM_ENVS"
echo "Iterations:         $MAX_ITERATIONS"
echo "Optimizer state:    reset (objective/distribution changed)"
echo "Iteration counter:  reset to 0 (new remediation stage)"
echo "Run name:           $RUN_NAME"
echo "Console log:        $console_log"
echo "GPU log:            $gpu_log"

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
    --resume_path "$SOURCE_CHECKPOINT"
    --reset_optimizer_on_resume
    --reset_iteration_on_resume
)

if [[ "$DRY_RUN" == "1" ]]; then
    printf 'Dry-run command:'
    printf ' %q' "${train_command[@]}"
    printf '\n'
    exit 0
fi

if [[ "$ENABLE_GPU_MONITOR" == "1" ]]; then
    fields="timestamp,index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
    nvidia-smi --query-gpu="$fields" --format=csv --loop="$GPU_POLL_SECONDS" >"$gpu_log" 2>&1 &
    gpu_monitor_pid=$!
    echo "GPU monitor PID: $gpu_monitor_pid"
fi

set +e
systemd-inhibit \
    --what=sleep \
    --why="Unitree Go2 v5 ascending remediation ${RUN_NAME}" \
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
