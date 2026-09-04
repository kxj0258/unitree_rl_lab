#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
CHECKPOINT="${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity_v5_stairs_specialist/2026-08-27_16-17-34_v5_perceptive_gru_stairs_specialist_formal_continue_from2200_trainseed42_env1024_addi2800_total5000/model_4999.pt"
EXPECTED_SHA256="47b42309c03450c88f27ceaf66ed393abfcc5a488e7a508a369055ed4cfcee50"
TRAINING_SEED=42
NUM_ENVS="${NUM_ENVS:-32}"
EPISODES_PER_ENV="${EPISODES_PER_ENV:-2}"
COMMAND_X="${COMMAND_X:-0.35}"
COMMAND_Y="${COMMAND_Y:-0.0}"
COMMAND_YAW="${COMMAND_YAW:-0.0}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_TAG="${OUTPUT_TAG:-2026-08-27_vx035_y0_yaw0_v1}"
DRY_RUN="${DRY_RUN:-0}"
EVALUATION_SEEDS=(7 42 123)

if [[ $# -ne 0 ]]; then
    echo "This wrapper takes no positional arguments." >&2
    exit 2
fi
if [[ "$NUM_ENVS" != "32" ]]; then
    echo "The fixed stairs protocol requires NUM_ENVS=32: $NUM_ENVS" >&2
    exit 2
fi
if [[ "$EPISODES_PER_ENV" != "2" ]]; then
    echo "The directional comparison requires EPISODES_PER_ENV=2: $EPISODES_PER_ENV" >&2
    exit 2
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1: $DRY_RUN" >&2
    exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi
actual_sha256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
    echo "Checkpoint SHA-256 mismatch" >&2
    echo "Expected: $EXPECTED_SHA256" >&2
    echo "Actual:   $actual_sha256" >&2
    exit 1
fi

cd "$PROJECT_DIR"
for evaluation_seed in "${EVALUATION_SEEDS[@]}"; do
    output_dir="${PROJECT_DIR}/evaluation/v5_stairs_directional_model4999_trainseed${TRAINING_SEED}_evalseed${evaluation_seed}_${OUTPUT_TAG}"
    console_log="${PROJECT_DIR}/../eval_v5_stairs_directional_model4999_trainseed${TRAINING_SEED}_evalseed${evaluation_seed}_${OUTPUT_TAG}.log"
    if [[ -e "$output_dir" ]]; then
        echo "Refusing to overwrite evaluation directory: $output_dir" >&2
        exit 1
    fi
    if [[ -e "$console_log" ]]; then
        echo "Refusing to overwrite console log: $console_log" >&2
        exit 1
    fi

    echo "Task:                     Unitree-Go2-Velocity-V5-Stairs-Specialist"
    echo "Task variant:             v5_stairs_specialist_directional"
    echo "Training seed:            $TRAINING_SEED"
    echo "Evaluation scenario seed: $evaluation_seed"
    echo "Checkpoint:               $CHECKPOINT"
    echo "Checkpoint SHA-256:       $actual_sha256"
    echo "Terrain protocol:         v5_stairs_stratified"
    echo "Fixed command:            vx=$COMMAND_X vy=$COMMAND_Y yaw=$COMMAND_YAW"
    echo "Environments:             $NUM_ENVS"
    echo "Episodes per environment: $EPISODES_PER_ENV"
    echo "Expected episodes:        $((NUM_ENVS * EPISODES_PER_ENV))"
    echo "Output directory:         $output_dir"
    echo "Console log:              $console_log"

    command=(
        python scripts/project/evaluate_go2.py
        --task Unitree-Go2-Velocity-V5-Stairs-Specialist
        --terrain_protocol v5_stairs_stratified
        --num_envs "$NUM_ENVS"
        --episodes_per_env "$EPISODES_PER_ENV"
        --seed "$evaluation_seed"
        --fixed_command_x "$COMMAND_X"
        --fixed_command_y "$COMMAND_Y"
        --fixed_command_yaw "$COMMAND_YAW"
        --checkpoint "$CHECKPOINT"
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
