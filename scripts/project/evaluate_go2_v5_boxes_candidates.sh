#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
RUN_DIR="${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity_v5_boxes_specialist/2026-08-28_02-49-20_v5_perceptive_gru_boxes_specialist_formal_v1_from_foundation3200_trainseed42_env1024_i4000"
TRAINING_SEED=42
NUM_ENVS="${NUM_ENVS:-32}"
EPISODES_PER_ENV="${EPISODES_PER_ENV:-2}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_TAG="${OUTPUT_TAG:-2026-08-28_boxes_selection_v1}"
DRY_RUN="${DRY_RUN:-0}"
EVALUATION_SEEDS=(7 42 123)
CHECKPOINT_LABELS=(3600 3800 3999)

declare -A EXPECTED_SHA256=(
    [3600]="5733f667d8189d0d9f9c4bf8f83b5bd32b969a2b83134b09f7e14c307a97be13"
    [3800]="03b38b0283e4c2339007bb3de7924810cc527b486b7032babeff1e3de35622cb"
    [3999]="4670abe7002b9667a16ef525dc6fd8057f101de7c848c6cf8cd87340721b7750"
)

if [[ $# -ne 0 ]]; then
    echo "This wrapper takes no positional arguments." >&2
    exit 2
fi
if [[ "$NUM_ENVS" != "32" ]]; then
    echo "The fixed boxes candidate protocol requires NUM_ENVS=32: $NUM_ENVS" >&2
    exit 2
fi
if [[ "$EPISODES_PER_ENV" != "2" ]]; then
    echo "The boxes candidate comparison requires EPISODES_PER_ENV=2: $EPISODES_PER_ENV" >&2
    exit 2
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1: $DRY_RUN" >&2
    exit 2
fi

cd "$PROJECT_DIR"

for checkpoint_label in "${CHECKPOINT_LABELS[@]}"; do
    checkpoint="${RUN_DIR}/model_${checkpoint_label}.pt"
    expected_sha256="${EXPECTED_SHA256[$checkpoint_label]}"
    if [[ ! -f "$checkpoint" ]]; then
        echo "Checkpoint not found: $checkpoint" >&2
        exit 1
    fi
    actual_sha256="$(sha256sum "$checkpoint" | awk '{print $1}')"
    if [[ "$actual_sha256" != "$expected_sha256" ]]; then
        echo "Checkpoint SHA-256 mismatch: $checkpoint" >&2
        echo "Expected: $expected_sha256" >&2
        echo "Actual:   $actual_sha256" >&2
        exit 1
    fi

    for evaluation_seed in "${EVALUATION_SEEDS[@]}"; do
        output_dir="${PROJECT_DIR}/evaluation/v5_boxes_stratified_model${checkpoint_label}_trainseed${TRAINING_SEED}_evalseed${evaluation_seed}_${OUTPUT_TAG}"
        console_log="${PROJECT_DIR}/../eval_v5_boxes_model${checkpoint_label}_trainseed${TRAINING_SEED}_evalseed${evaluation_seed}_${OUTPUT_TAG}.log"
        if [[ -e "$output_dir" ]]; then
            echo "Refusing to overwrite evaluation directory: $output_dir" >&2
            exit 1
        fi
        if [[ -e "$console_log" ]]; then
            echo "Refusing to overwrite console log: $console_log" >&2
            exit 1
        fi

        echo "Task:                     Unitree-Go2-Velocity-V5-Boxes-Specialist"
        echo "Task variant:             v5_boxes_specialist"
        echo "Training seed:            $TRAINING_SEED"
        echo "Evaluation scenario seed: $evaluation_seed"
        echo "Source checkpoint:        foundation model_3200.pt"
        echo "Current checkpoint:       model_${checkpoint_label}.pt"
        echo "Checkpoint:               $checkpoint"
        echo "Checkpoint SHA-256:       $actual_sha256"
        echo "Terrain protocol:         v5_boxes_stratified"
        echo "Environments:             $NUM_ENVS"
        echo "Episodes per environment: $EPISODES_PER_ENV"
        echo "Expected episodes:        $((NUM_ENVS * EPISODES_PER_ENV))"
        echo "Terrain coverage:         boxes only, levels 0-7, 4 envs per level"
        echo "Command coverage:         seeded random training distribution"
        echo "Output directory:         $output_dir"
        echo "Console log:              $console_log"

        command=(
            python scripts/project/evaluate_go2.py
            --task Unitree-Go2-Velocity-V5-Boxes-Specialist
            --terrain_protocol v5_boxes_stratified
            --num_envs "$NUM_ENVS"
            --episodes_per_env "$EPISODES_PER_ENV"
            --seed "$evaluation_seed"
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
done
