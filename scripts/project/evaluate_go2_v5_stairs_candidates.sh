#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
ORIGINAL_RUN="${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity_v5_stairs_specialist/2026-08-27_11-22-47_v5_perceptive_gru_stairs_specialist_formal_from_foundation3200_trainseed42_env1024_i5000"
CONTINUATION_RUN="${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity_v5_stairs_specialist/2026-08-27_16-17-34_v5_perceptive_gru_stairs_specialist_formal_continue_from2200_trainseed42_env1024_addi2800_total5000"
TRAINING_SEED=42
NUM_ENVS="${NUM_ENVS:-32}"
EPISODES_PER_ENV="${EPISODES_PER_ENV:-2}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_TAG="${OUTPUT_TAG:-2026-08-27_stairs_selection_v1}"
DRY_RUN="${DRY_RUN:-0}"
EVALUATION_SEEDS=(7 42 123)
CHECKPOINT_LABELS=(1200 2200 3500 4500 4999)
COMPLETED_SMOKE_DIR="${PROJECT_DIR}/evaluation/v5_stairs_stratified_model4999_trainseed42_evalseed42_2026-08-27_protocol_smoke_v1"

declare -A CHECKPOINT_PATHS=(
    [1200]="${ORIGINAL_RUN}/model_1200.pt"
    [2200]="${ORIGINAL_RUN}/model_2200.pt"
    [3500]="${CONTINUATION_RUN}/model_3500.pt"
    [4500]="${CONTINUATION_RUN}/model_4500.pt"
    [4999]="${CONTINUATION_RUN}/model_4999.pt"
)
declare -A EXPECTED_SHA256=(
    [1200]="ee7336cc72d13b96d278cfcfc1b5ab81b86338c268f1bfc6a0d8cea32a98da00"
    [2200]="db25354aadfaebc66f32ab0ee42a7bff28d15b2ae1d168901f959cebedbdc345"
    [3500]="8e84a27a51dd080195bf9a1298064d8f8c9980d7ec389a7faf3fb7b729ead3ee"
    [4500]="c2f3ff6cde8e24c773aaab22ab9bb608281dfd180925fd41bdbcf83bd70b0038"
    [4999]="47b42309c03450c88f27ceaf66ed393abfcc5a488e7a508a369055ed4cfcee50"
)

if [[ $# -ne 0 ]]; then
    echo "This wrapper takes no positional arguments." >&2
    exit 2
fi
if [[ "$NUM_ENVS" != "32" ]]; then
    echo "The fixed stairs protocol requires NUM_ENVS=32: $NUM_ENVS" >&2
    exit 2
fi
if [[ "$EPISODES_PER_ENV" != "2" ]]; then
    echo "The candidate comparison requires EPISODES_PER_ENV=2: $EPISODES_PER_ENV" >&2
    exit 2
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1: $DRY_RUN" >&2
    exit 2
fi

cd "$PROJECT_DIR"

for checkpoint_label in "${CHECKPOINT_LABELS[@]}"; do
    checkpoint="${CHECKPOINT_PATHS[$checkpoint_label]}"
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
        if [[ "$checkpoint_label" == "4999" && "$evaluation_seed" == "42" ]]; then
            echo "Reusing completed protocol validation instead of repeating model4999/evalseed42:"
            echo "  $COMPLETED_SMOKE_DIR"
            (
                cd "$COMPLETED_SMOKE_DIR"
                echo "9a6c12c55a82e0120c5fc4f4d3078be7a9e22fce4af0e6a29dbe1dc393dcf815  metadata.json" | sha256sum -c -
                echo "e4fd74c717136f24ec6aeeae8c5d3d046c51f919a6cd7dc71448e6e7de9e8239  episodes.csv" | sha256sum -c -
                echo "b4d0648261124bcde4a284d22e28046e94df83b32cc69401e231072972345e13  summary.json" | sha256sum -c -
            )
            continue
        fi
        output_dir="${PROJECT_DIR}/evaluation/v5_stairs_stratified_model${checkpoint_label}_trainseed${TRAINING_SEED}_evalseed${evaluation_seed}_${OUTPUT_TAG}"
        console_log="${PROJECT_DIR}/../eval_v5_stairs_model${checkpoint_label}_trainseed${TRAINING_SEED}_evalseed${evaluation_seed}_${OUTPUT_TAG}.log"
        if [[ -e "$output_dir" ]]; then
            echo "Refusing to overwrite evaluation directory: $output_dir" >&2
            exit 1
        fi
        if [[ -e "$console_log" ]]; then
            echo "Refusing to overwrite console log: $console_log" >&2
            exit 1
        fi

        echo "Task:                     Unitree-Go2-Velocity-V5-Stairs-Specialist"
        echo "Task variant:             v5_stairs_specialist"
        echo "Training seed:            $TRAINING_SEED"
        echo "Evaluation scenario seed: $evaluation_seed"
        echo "Checkpoint label:         model_${checkpoint_label}.pt"
        echo "Checkpoint:               $checkpoint"
        echo "Checkpoint SHA-256:       $actual_sha256"
        echo "Terrain protocol:         v5_stairs_stratified"
        echo "Environments:             $NUM_ENVS"
        echo "Episodes per environment: $EPISODES_PER_ENV"
        echo "Expected episodes:        $((NUM_ENVS * EPISODES_PER_ENV))"
        echo "Terrain coverage:         4 types x 8 levels"
        echo "Output directory:         $output_dir"
        echo "Console log:              $console_log"

        command=(
            python scripts/project/evaluate_go2.py
            --task Unitree-Go2-Velocity-V5-Stairs-Specialist
            --terrain_protocol v5_stairs_stratified
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
