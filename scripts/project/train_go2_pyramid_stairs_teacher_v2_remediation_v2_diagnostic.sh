#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
PYTHON_EXE="${PYTHON_EXE:-/home/kxj/miniforge3/envs/unitree_rl/bin/python}"
NUM_ENVS="${NUM_ENVS:-256}"
TRAINING_SEED="${TRAINING_SEED:-42}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1000}"
DEVICE="${DEVICE:-cuda:0}"
LOGGER="${LOGGER:-tensorboard}"
ENABLE_GPU_MONITOR="${ENABLE_GPU_MONITOR:-1}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-5}"
DRY_RUN="${DRY_RUN:-0}"
RUN_STAGE="${RUN_STAGE:-diagnostic}"

TASK="Unitree-Go2-Velocity-TeacherV2-PyramidStairsRemediationV2"
EXPERIMENT_NAME="unitree_go2_velocity_teacher_v2_pyramid_stairs_remediation_v2"
RUN_NAME="teacher_v2_pyramid_stairs_remediation_v2_gaitclock_participation_fromscratch_${RUN_STAGE}_trainseed${TRAINING_SEED}_env${NUM_ENVS}_i${MAX_ITERATIONS}"
CONSOLE_ROOT="${CONSOLE_ROOT:-${PROJECT_DIR}/logs/teacher_v2_console}"
console_log="${CONSOLE_ROOT}/${RUN_NAME}.console.log"
gpu_log="${CONSOLE_ROOT}/${RUN_NAME}.gpu.csv"
experiment_root="${PROJECT_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}"
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
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "$name must be positive" >&2; exit 2; }
done
[[ "$TRAINING_SEED" =~ ^[0-9]+$ ]] || { echo "TRAINING_SEED must be non-negative" >&2; exit 2; }
[[ "$ENABLE_GPU_MONITOR" =~ ^[01]$ ]] || { echo "ENABLE_GPU_MONITOR must be 0 or 1" >&2; exit 2; }
[[ "$DRY_RUN" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
[[ "$RUN_STAGE" == "diagnostic" || "$RUN_STAGE" == "formal" ]] || {
    echo "RUN_STAGE must be diagnostic or formal" >&2
    exit 2
}
[[ -d "$PROJECT_DIR" && -x "$PYTHON_EXE" ]] || { echo "Project or Python missing" >&2; exit 1; }

if find "$experiment_root" -mindepth 1 -maxdepth 1 -type d -name "*_${RUN_NAME}" -print -quit 2>/dev/null | grep -q .; then
    echo "Refusing to repeat existing diagnostic run: $RUN_NAME" >&2
    exit 1
fi
[[ ! -e "$console_log" && ! -e "$gpu_log" ]] || { echo "Refusing to overwrite diagnostic logs" >&2; exit 1; }

train_command=(
    ./unitree_rl_lab.sh --train --task "$TASK" --num_envs "$NUM_ENVS"
    --seed "$TRAINING_SEED" --max_iterations "$MAX_ITERATIONS"
    --run_name "$RUN_NAME" --experiment_name "$EXPERIMENT_NAME"
    --logger "$LOGGER" --device "$DEVICE"
)

echo "Task:                     $TASK"
echo "Task variant:             pyramid_stairs_remediation_v2_gaitclock_participation"
if [[ "$RUN_STAGE" == "formal" ]]; then
    echo "Run scope:                formal teacher candidate training"
else
    echo "Run scope:                bounded diagnostic; not a final teacher"
fi
echo "Initialization:           random/from-scratch (resume forbidden)"
echo "Run name:                 $RUN_NAME"
echo "Training seed:            $TRAINING_SEED"
echo "Evaluation scenario seed: none"
echo "Source checkpoint:        none"
echo "Command contract:         vx=0.15..0.50, vy=0, relative heading target +/-45deg"
echo "Terrain curriculum:       pyramid stairs 0.02..0.14m, bilateral-gait-gated"
echo "Experiment root:          $experiment_root"
echo "Console log:              $console_log"
echo "GPU log:                  $gpu_log"

if [[ "$DRY_RUN" == 1 ]]; then
    printf 'Dry-run command:'; printf ' %q' "${train_command[@]}"; printf '\n'
    exit 0
fi

mkdir -p "$CONSOLE_ROOT"
cd "$PROJECT_DIR"
if [[ "$ENABLE_GPU_MONITOR" == 1 ]]; then
    nvidia-smi --query-gpu=timestamp,index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
        --format=csv --loop="$GPU_POLL_SECONDS" >"$gpu_log" 2>&1 &
    gpu_monitor_pid=$!
fi

set +e
systemd-inhibit --what=sleep --why="Go2 pyramid stairs Teacher v2 remediation-v2 $RUN_STAGE $RUN_NAME" \
    "${train_command[@]}" 2>&1 | tee "$console_log"
training_status=${PIPESTATUS[0]}
set -e
cleanup
gpu_monitor_pid=""
echo "Pyramid Stairs Teacher v2 remediation-v2 $RUN_STAGE exit code: $training_status"
exit "$training_status"
