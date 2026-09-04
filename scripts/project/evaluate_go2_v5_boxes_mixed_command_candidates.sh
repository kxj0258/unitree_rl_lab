#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
EVALUATION_VARIANT="${EVALUATION_VARIANT:-mixed_command}"
TRAINING_SEED=42
EVALUATION_SEED="${EVALUATION_SEED:-42}"
NUM_ENVS=8
EPISODES_PER_ENV=2
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_TAG="${OUTPUT_TAG:-2026-08-28_boxes_mixed_command_candidates_v2}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a DIRECTION_LABELS <<< "${DIRECTION_LABELS_OVERRIDE:-forward reverse lateral_left forward_yaw}"

declare -A EXPECTED_SHA256
case "$EVALUATION_VARIANT" in
    mixed_command)
        RUN_DIR="${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity_v5_boxes_mixed_command_remediation/2026-08-28_15-23-21_v5_boxes_mixed_command_remediation_formal_v1_from_boxes3999_trainseed42_env1024_i2000"
        TASK="Unitree-Go2-Velocity-V5-Boxes-Mixed-Command-Remediation"
        TASK_VARIANT="v5_boxes_mixed_command_remediation"
        SOURCE_DESCRIPTION="boxes specialist model_3999.pt"
        OUTPUT_PREFIX="v5_boxes_mixed_command"
        read -r -a CHECKPOINT_LABELS <<< "${CHECKPOINT_LABELS_OVERRIDE:-500 1000 1500 1999}"
        EXPECTED_SHA256[500]="520ded689f56f09d8543db7fb23e46dc67c7012046fdb75fb9f821a553d0ef23"
        EXPECTED_SHA256[1000]="42833bbde6de382ed89a38d1d986049ec4ef1552da400da446720aa512c73a28"
        EXPECTED_SHA256[1500]="eae887bda6365ebc18608a9888c5e55d4deba8736f40f4d51950c5805a8c9da3"
        EXPECTED_SHA256[1999]="cce1f636b085409124b3526f63a785131a40e12cb9d73a63a655f4e265f8016b"
        ;;
    safety_remediation)
        RUN_DIR="${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity_v5_boxes_safety_remediation/2026-08-28_20-41-58_v5_boxes_safety_remediation_v2_formal_from_mixed1999_trainseed42_env1024_i1000"
        TASK="Unitree-Go2-Velocity-V5-Boxes-Safety-Remediation"
        TASK_VARIANT="v5_boxes_safety_remediation_v2"
        SOURCE_DESCRIPTION="mixed-command remediation model_1999.pt"
        OUTPUT_PREFIX="v5_boxes_safety_remediation"
        read -r -a CHECKPOINT_LABELS <<< "${CHECKPOINT_LABELS_OVERRIDE:-500 700 900 999}"
        EXPECTED_SHA256[500]="8554827194454840eece63f6ede37de4f577d53083e6af7984821397d5babae9"
        EXPECTED_SHA256[700]="74d959f20d931373c2507345188a1d307a1b1611c23f88d7330951cfe0cd4faa"
        EXPECTED_SHA256[900]="79f98b9780be4936f5fbd199147bf5865d5567a9184c4a75f82bb245098ea3cb"
        EXPECTED_SHA256[999]="ebfa11d9e1eb188c27fbae76e1b38c1ab2ec3e009a7466e60ea5b675f265f32a"
        ;;
    *) echo "Unsupported EVALUATION_VARIANT: $EVALUATION_VARIANT" >&2; exit 2 ;;
esac
declare -A COMMAND_X=(
    [forward]="0.35" [reverse]="-0.30" [lateral_left]="0.00" [lateral_right]="0.00"
    [forward_yaw]="0.30" [reverse_yaw]="-0.25"
)
declare -A COMMAND_Y=(
    [forward]="0.00" [reverse]="0.00" [lateral_left]="0.25" [lateral_right]="-0.25"
    [forward_yaw]="0.00" [reverse_yaw]="0.00"
)
declare -A COMMAND_YAW=(
    [forward]="0.00" [reverse]="0.00" [lateral_left]="0.00" [lateral_right]="0.00"
    [forward_yaw]="0.35" [reverse_yaw]="-0.30"
)

if [[ $# -ne 0 ]]; then
    echo "This wrapper takes no positional arguments." >&2
    exit 2
fi
if [[ ! "$EVALUATION_SEED" =~ ^[0-9]+$ ]]; then
    echo "EVALUATION_SEED must be a non-negative integer: $EVALUATION_SEED" >&2
    exit 2
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1: $DRY_RUN" >&2
    exit 2
fi
for checkpoint_label in "${CHECKPOINT_LABELS[@]}"; do
    if [[ -z "${EXPECTED_SHA256[$checkpoint_label]:-}" ]]; then
        echo "Unsupported checkpoint label: $checkpoint_label" >&2
        exit 2
    fi
done
for direction in "${DIRECTION_LABELS[@]}"; do
    if [[ -z "${COMMAND_X[$direction]:-}" ]]; then
        echo "Unsupported direction label: $direction" >&2
        exit 2
    fi
done

cd "$PROJECT_DIR"
for checkpoint_label in "${CHECKPOINT_LABELS[@]}"; do
    checkpoint="${RUN_DIR}/model_${checkpoint_label}.pt"
    expected_sha256="${EXPECTED_SHA256[$checkpoint_label]}"
    [[ -f "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; exit 1; }
    actual_sha256="$(sha256sum "$checkpoint" | awk '{print $1}')"
    if [[ "$actual_sha256" != "$expected_sha256" ]]; then
        echo "Checkpoint SHA-256 mismatch: $checkpoint" >&2
        echo "Expected: $expected_sha256" >&2
        echo "Actual:   $actual_sha256" >&2
        exit 1
    fi

    for direction in "${DIRECTION_LABELS[@]}"; do
        output_dir="${PROJECT_DIR}/evaluation/${OUTPUT_PREFIX}_model${checkpoint_label}_trainseed${TRAINING_SEED}_evalseed${EVALUATION_SEED}_${direction}_${OUTPUT_TAG}"
        console_log="${PROJECT_DIR}/../eval_${OUTPUT_PREFIX}_model${checkpoint_label}_trainseed${TRAINING_SEED}_evalseed${EVALUATION_SEED}_${direction}_${OUTPUT_TAG}.log"
        [[ ! -e "$output_dir" ]] || { echo "Refusing to overwrite: $output_dir" >&2; exit 1; }
        [[ ! -e "$console_log" ]] || { echo "Refusing to overwrite: $console_log" >&2; exit 1; }

        echo "Task:                     $TASK"
        echo "Task variant:             $TASK_VARIANT"
        echo "Training seed:            $TRAINING_SEED"
        echo "Evaluation scenario seed: $EVALUATION_SEED"
        echo "Source checkpoint:        $SOURCE_DESCRIPTION"
        echo "Current checkpoint:       model_${checkpoint_label}.pt"
        echo "Checkpoint:               $checkpoint"
        echo "Checkpoint SHA-256:       $actual_sha256"
        echo "Terrain protocol:         v5_boxes_stratified"
        echo "Terrain coverage:         boxes only, levels 0-7"
        echo "Direction label:          $direction"
        echo "Fixed command:            vx=${COMMAND_X[$direction]}, vy=${COMMAND_Y[$direction]}, yaw_rate=${COMMAND_YAW[$direction]}"
        echo "Expected episodes:        $((NUM_ENVS * EPISODES_PER_ENV))"
        echo "Output directory:         $output_dir"
        echo "Console log:              $console_log"

        command=(python scripts/project/evaluate_go2.py
            --task "$TASK" --terrain_protocol v5_boxes_stratified
            --num_envs "$NUM_ENVS" --episodes_per_env "$EPISODES_PER_ENV"
            --seed "$EVALUATION_SEED"
            --fixed_command_x "${COMMAND_X[$direction]}"
            --fixed_command_y "${COMMAND_Y[$direction]}"
            --fixed_command_yaw "${COMMAND_YAW[$direction]}"
            --checkpoint "$checkpoint" --output_dir "$output_dir" --device "$DEVICE")

        if [[ "$DRY_RUN" == "1" ]]; then
            printf 'Dry-run command:'; printf ' %q' "${command[@]}"; printf '\n'
            continue
        fi
        set +e
        "${command[@]}" 2>&1 | tee "$console_log"
        evaluation_status=${PIPESTATUS[0]}
        set -e
        echo "Evaluation exit code: $evaluation_status"
        [[ "$evaluation_status" -eq 0 ]] || exit "$evaluation_status"
    done
done
