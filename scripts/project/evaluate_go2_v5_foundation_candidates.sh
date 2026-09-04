#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
RUN_DIR="${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity_v5_foundation/2026-08-26_23-13-40_multiterrain_v5_perceptive_gru_foundation_balanced4_fromscratch_seed42_env1024_i5000"
TRAINING_SEED=42
EVALUATION_SEED="${EVALUATION_SEED:-42}"
NUM_ENVS="${NUM_ENVS:-32}"
EPISODES_PER_ENV="${EPISODES_PER_ENV:-2}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DATE="${OUTPUT_DATE:-2026-08-27}"
DRY_RUN="${DRY_RUN:-0}"
CHECKPOINTS=(2100 3200 4999)

if [[ $# -ne 0 ]]; then
    echo "This wrapper takes no positional arguments." >&2
    exit 2
fi
if [[ ! "$EVALUATION_SEED" =~ ^[0-9]+$ ]]; then
    echo "EVALUATION_SEED must be a non-negative integer: $EVALUATION_SEED" >&2
    exit 2
fi
if [[ "$NUM_ENVS" != "32" ]]; then
    echo "This fixed protocol requires NUM_ENVS=32: $NUM_ENVS" >&2
    exit 2
fi
if [[ "$EPISODES_PER_ENV" != "2" ]]; then
    echo "This candidate comparison requires EPISODES_PER_ENV=2: $EPISODES_PER_ENV" >&2
    exit 2
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1: $DRY_RUN" >&2
    exit 2
fi

cd "$PROJECT_DIR"

for iteration in "${CHECKPOINTS[@]}"; do
    checkpoint="${RUN_DIR}/model_${iteration}.pt"
    output_dir="${PROJECT_DIR}/evaluation/v5_foundation_stratified_model${iteration}_trainseed${TRAINING_SEED}_evalseed${EVALUATION_SEED}_${OUTPUT_DATE}"
    console_log="${PROJECT_DIR}/../eval_v5_foundation_model${iteration}_trainseed${TRAINING_SEED}_evalseed${EVALUATION_SEED}_${OUTPUT_DATE}.log"

    if [[ ! -f "$checkpoint" ]]; then
        echo "Checkpoint not found: $checkpoint" >&2
        exit 1
    fi
    if [[ -e "$output_dir" ]]; then
        echo "Refusing to overwrite evaluation directory: $output_dir" >&2
        exit 1
    fi
    if [[ -e "$console_log" ]]; then
        echo "Refusing to overwrite console log: $console_log" >&2
        exit 1
    fi

    echo "Task:                     Unitree-Go2-Velocity-V5-Foundation"
    echo "Training seed:            $TRAINING_SEED"
    echo "Evaluation scenario seed: $EVALUATION_SEED"
    echo "Checkpoint:               $checkpoint"
    echo "Checkpoint SHA-256:       $(sha256sum "$checkpoint" | awk '{print $1}')"
    echo "Terrain protocol:         v5_foundation_stratified"
    echo "Environments:             $NUM_ENVS"
    echo "Episodes per environment: $EPISODES_PER_ENV"
    echo "Expected episodes:        $((NUM_ENVS * EPISODES_PER_ENV))"
    echo "Output directory:         $output_dir"
    echo "Console log:              $console_log"

    command=(
        python scripts/project/evaluate_go2.py
        --task Unitree-Go2-Velocity-V5-Foundation
        --terrain_protocol v5_foundation_stratified
        --num_envs "$NUM_ENVS"
        --episodes_per_env "$EPISODES_PER_ENV"
        --seed "$EVALUATION_SEED"
        --checkpoint "$checkpoint"
        --output_dir "$output_dir"
        --device "$DEVICE"
    )

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'Dry-run command:'
        printf ' %q' "${command[@]}"
        printf '\n'
        continue
    fi

    set +e
    "${command[@]}" 2>&1 | tee "$console_log"
    evaluation_status=${PIPESTATUS[0]}
    set -e
    echo "Evaluation exit code: $evaluation_status"
    if [[ "$evaluation_status" -ne 0 ]]; then
        exit "$evaluation_status"
    fi
done
