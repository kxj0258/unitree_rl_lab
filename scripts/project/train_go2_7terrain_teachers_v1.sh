#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
NUM_ENVS="${NUM_ENVS:-1024}"
TRAINING_SEED="${TRAINING_SEED:-42}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
DEVICE="${DEVICE:-cuda:0}"
LOGGER="${LOGGER:-tensorboard}"
ENABLE_GPU_MONITOR="${ENABLE_GPU_MONITOR:-1}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-5}"
DRY_RUN="${DRY_RUN:-0}"

terrains=(
    flat
    random_rough
    pyramid_slope
    inverted_pyramid_slope
    pyramid_stairs
    inverted_pyramid_stairs
    boxes
)

[[ -d "$PROJECT_DIR" ]] || { echo "Project directory not found: $PROJECT_DIR" >&2; exit 1; }
cd "$PROJECT_DIR"

# Complete preflight before launching the first expensive run.  This prevents
# discovering a duplicate namespace only after earlier teachers have trained.
for terrain in "${terrains[@]}"; do
    TERRAIN="$terrain" NUM_ENVS="$NUM_ENVS" TRAINING_SEED="$TRAINING_SEED" \
        MAX_ITERATIONS="$MAX_ITERATIONS" DEVICE="$DEVICE" LOGGER="$LOGGER" \
        ENABLE_GPU_MONITOR="$ENABLE_GPU_MONITOR" GPU_POLL_SECONDS="$GPU_POLL_SECONDS" \
        DRY_RUN=1 scripts/project/train_go2_terrain_teacher_v1.sh
done

if [[ "$DRY_RUN" == 1 ]]; then
    echo "Seven-teacher preflight completed; no training launched."
    exit 0
fi

for terrain in "${terrains[@]}"; do
    echo "Starting terrain teacher: $terrain"
    TERRAIN="$terrain" NUM_ENVS="$NUM_ENVS" TRAINING_SEED="$TRAINING_SEED" \
        MAX_ITERATIONS="$MAX_ITERATIONS" DEVICE="$DEVICE" LOGGER="$LOGGER" \
        ENABLE_GPU_MONITOR="$ENABLE_GPU_MONITOR" GPU_POLL_SECONDS="$GPU_POLL_SECONDS" \
        DRY_RUN=0 scripts/project/train_go2_terrain_teacher_v1.sh
    echo "Completed terrain teacher: $terrain"
done

echo "All seven terrain-teacher training processes exited successfully."
