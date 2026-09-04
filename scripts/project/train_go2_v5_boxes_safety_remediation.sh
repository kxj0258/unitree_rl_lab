#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"
TASK="Unitree-Go2-Velocity-V5-Boxes-Safety-Remediation"
EXPERIMENT_NAME="unitree_go2_velocity_v5_boxes_safety_remediation"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity_v5_boxes_mixed_command_remediation/2026-08-28_15-23-21_v5_boxes_mixed_command_remediation_formal_v1_from_boxes3999_trainseed42_env1024_i2000/model_1999.pt}"
EXPECTED_SOURCE_SHA256="cce1f636b085409124b3526f63a785131a40e12cb9d73a63a655f4e265f8016b"
CONTINUE_CHECKPOINT="${CONTINUE_CHECKPOINT:-}"
EXPECTED_CONTINUE_SHA256="${EXPECTED_CONTINUE_SHA256:-}"

NUM_ENVS="${NUM_ENVS:-128}"
SEED="${SEED:-42}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5}"
RUN_STAGE="${RUN_STAGE:-smoke}"
LOGGER="${LOGGER:-tensorboard}"
DEVICE="${DEVICE:-cuda:0}"
ENABLE_GPU_MONITOR="${ENABLE_GPU_MONITOR:-1}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-5}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -n "$CONTINUE_CHECKPOINT" ]]; then
    resume_checkpoint="$CONTINUE_CHECKPOINT"
    expected_resume_sha256="$EXPECTED_CONTINUE_SHA256"
    resume_mode="continue safety remediation with optimizer and iteration state"
    reset_optimizer=0
    reset_iteration=0
    source_tag="continue_safety_remediation"
else
    resume_checkpoint="$SOURCE_CHECKPOINT"
    expected_resume_sha256="$EXPECTED_SOURCE_SHA256"
    resume_mode="transfer mixed-command model1999 weights"
    reset_optimizer=1
    reset_iteration=1
    source_tag="from_mixed1999"
fi

RUN_NAME="${V5_BOXES_SAFETY_RUN_NAME:-v5_boxes_safety_remediation_${RUN_STAGE}_${source_tag}_trainseed${SEED}_env${NUM_ENVS}_i${MAX_ITERATIONS}}"
timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
console_log="${V5_BOXES_SAFETY_CONSOLE_LOG:-${WORKSPACE_DIR}/unitree_go2_${RUN_NAME}_${timestamp}.log}"
gpu_log="${V5_BOXES_SAFETY_GPU_LOG:-${WORKSPACE_DIR}/gpu_${RUN_NAME}_${timestamp}.csv}"
log_root="${PROJECT_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}"
gpu_monitor_pid=""

cleanup() {
    if [[ -n "$gpu_monitor_pid" ]] && kill -0 "$gpu_monitor_pid" 2>/dev/null; then
        kill "$gpu_monitor_pid" 2>/dev/null || true
        wait "$gpu_monitor_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

[[ $# -eq 0 ]] || { echo "This wrapper takes no positional arguments." >&2; exit 2; }
for name in NUM_ENVS MAX_ITERATIONS GPU_POLL_SECONDS; do
    value="${!name}"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "$name must be a positive integer: $value" >&2; exit 2; }
done
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be non-negative: $SEED" >&2; exit 2; }
[[ "$ENABLE_GPU_MONITOR" =~ ^[01]$ ]] || { echo "ENABLE_GPU_MONITOR must be 0 or 1" >&2; exit 2; }
[[ "$DRY_RUN" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
if [[ -n "$CONTINUE_CHECKPOINT" && -z "$EXPECTED_CONTINUE_SHA256" ]]; then
    echo "EXPECTED_CONTINUE_SHA256 is required with CONTINUE_CHECKPOINT" >&2; exit 2
fi
[[ -f "$resume_checkpoint" ]] || { echo "Resume checkpoint not found: $resume_checkpoint" >&2; exit 1; }
resume_sha256="$(sha256sum "$resume_checkpoint" | awk '{print $1}')"
if [[ "$resume_sha256" != "$expected_resume_sha256" ]]; then
    echo "Resume checkpoint SHA-256 mismatch." >&2
    echo "Expected: $expected_resume_sha256" >&2
    echo "Actual:   $resume_sha256" >&2
    exit 1
fi
if find "$log_root" -mindepth 1 -maxdepth 1 -type d -name "*_${RUN_NAME}" -print -quit 2>/dev/null | grep -q .; then
    echo "Refusing to reuse an existing safety-remediation run name: $RUN_NAME" >&2; exit 1
fi
if [[ -e "$console_log" || ( "$ENABLE_GPU_MONITOR" == "1" && -e "$gpu_log" ) ]]; then
    echo "Refusing to overwrite an existing console or GPU log." >&2; exit 1
fi
if [[ "$ENABLE_GPU_MONITOR" == "1" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required when ENABLE_GPU_MONITOR=1" >&2; exit 1
fi

cd "$PROJECT_DIR"
echo "Task:               $TASK"
echo "Task variant:       v5_boxes_safety_remediation_v2"
echo "Resume mode:        $resume_mode"
echo "Source checkpoint:  $resume_checkpoint"
echo "Source SHA256:      $resume_sha256"
if [[ "$reset_optimizer" == "1" ]]; then
    echo "Optimizer state:    reset (safety objective and distribution changed)"
    echo "Iteration counter:  reset to 0"
else
    echo "Optimizer state:    loaded from safety-remediation checkpoint"
    echo "Iteration counter:  continued from safety-remediation checkpoint"
fi
echo "Commands:           vx=[-0.40,0.40], vy=[-0.25,0.20], yaw=[-0.40,0.35]"
echo "Terrains:           90% boxes + 5% flat + 5% rough; fixed uniform levels 0-7"
echo "Safety objective:   base contact + orientation + stall + stumble"
echo "Experiment:         $EXPERIMENT_NAME"
echo "Environments:       $NUM_ENVS"
echo "Training seed:      $SEED"
echo "Eval scenario seed: N/A (training command)"
echo "Iterations:         $MAX_ITERATIONS"
echo "Run stage:          $RUN_STAGE"
echo "Run name:           $RUN_NAME"
echo "Device:             $DEVICE"
echo "Git HEAD:           $(git rev-parse HEAD)"
echo "Console log:        $console_log"
[[ "$ENABLE_GPU_MONITOR" == "1" ]] && echo "GPU log:            $gpu_log" || echo "GPU log:            disabled"

train_command=(./unitree_rl_lab.sh --train --task "$TASK" --num_envs "$NUM_ENVS" --seed "$SEED"
    --max_iterations "$MAX_ITERATIONS" --run_name "$RUN_NAME" --experiment_name "$EXPERIMENT_NAME"
    --logger "$LOGGER" --device "$DEVICE" --resume --resume_path "$resume_checkpoint")
[[ "$reset_optimizer" == "1" ]] && train_command+=(--reset_optimizer_on_resume)
[[ "$reset_iteration" == "1" ]] && train_command+=(--reset_iteration_on_resume)

if [[ "$DRY_RUN" == "1" ]]; then
    printf 'Dry-run command:'; printf ' %q' "${train_command[@]}"; printf '\n'; exit 0
fi
if [[ "$ENABLE_GPU_MONITOR" == "1" ]]; then
    fields="timestamp,index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
    nvidia-smi --query-gpu="$fields" --format=csv --loop="$GPU_POLL_SECONDS" >"$gpu_log" 2>&1 &
    gpu_monitor_pid=$!
    echo "GPU monitor PID: $gpu_monitor_pid"
fi
set +e
systemd-inhibit --what=sleep --why="Unitree Go2 boxes safety remediation ${RUN_NAME}" \
    "${train_command[@]}" 2>&1 | tee "$console_log"
training_status=${PIPESTATUS[0]}
set -e
cleanup; gpu_monitor_pid=""
echo "Training exit code: $training_status"
echo "Console log: $console_log"
[[ "$ENABLE_GPU_MONITOR" == "1" ]] && echo "GPU log: $gpu_log"
exit "$training_status"
