#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"

TASK="${TASK:-Unitree-Go2-Velocity}"
NUM_ENVS="${NUM_ENVS:-2048}"
SEED="${SEED:-42}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3000}"
RUN_NAME="${RUN_NAME:-baseline_seed${SEED}_env${NUM_ENVS}_i${MAX_ITERATIONS}}"
LOGGER="${LOGGER:-tensorboard}"

timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
console_log="${CONSOLE_LOG:-${WORKSPACE_DIR}/unitree_go2_${RUN_NAME}_${timestamp}.log}"

cd "$PROJECT_DIR"

if [[ -e "$console_log" ]]; then
    echo "Refusing to overwrite existing console log: $console_log" >&2
    exit 1
fi

echo "Task:           $TASK"
echo "Environments:   $NUM_ENVS"
echo "Seed:           $SEED"
echo "Iterations:     $MAX_ITERATIONS"
echo "Run name:       $RUN_NAME"
echo "Console log:    $console_log"

set +e
systemd-inhibit \
    --what=sleep \
    --why="Unitree Go2 training ${RUN_NAME}" \
    ./unitree_rl_lab.sh --train \
    --task "$TASK" \
    --num_envs "$NUM_ENVS" \
    --seed "$SEED" \
    --max_iterations "$MAX_ITERATIONS" \
    --run_name "$RUN_NAME" \
    --logger "$LOGGER" \
    2>&1 | tee "$console_log"

training_status=${PIPESTATUS[0]}
set -e

echo "Training exit code: $training_status"
exit "$training_status"
