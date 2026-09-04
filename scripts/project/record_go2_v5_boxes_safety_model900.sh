#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
PYTHON_EXE="${PYTHON_EXE:-/home/kxj/miniforge3/envs/unitree_rl/bin/python}"
TASK="Unitree-Go2-Velocity-V5-Boxes-Safety-Remediation"
CHECKPOINT="${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity_v5_boxes_safety_remediation/2026-08-28_20-41-58_v5_boxes_safety_remediation_v2_formal_from_mixed1999_trainseed42_env1024_i1000/model_900.pt"
EXPECTED_SHA256="79f98b9780be4936f5fbd199147bf5865d5567a9184c4a75f82bb245098ea3cb"
TRAINING_SEED=42
EVALUATION_SEED="${EVALUATION_SEED:-42}"
TERRAIN_LEVEL="${TERRAIN_LEVEL:-7}"
VIDEO_LENGTH="${VIDEO_LENGTH:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
CAMERA_MODE="${CAMERA_MODE:-asset_follow}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/evaluation/video_v5_boxes_safety_model900_trainseed42_evalseed${EVALUATION_SEED}_level${TERRAIN_LEVEL}_sixaxis_evidence_v1}"
DRY_RUN="${DRY_RUN:-0}"

declare -a LABELS=(forward lateral_right reverse_yaw)
declare -A COMMAND_X=([forward]="0.35" [lateral_right]="0.00" [reverse_yaw]="-0.25")
declare -A COMMAND_Y=([forward]="0.00" [lateral_right]="-0.25" [reverse_yaw]="0.00")
declare -A COMMAND_YAW=([forward]="0.00" [lateral_right]="0.00" [reverse_yaw]="-0.30")

[[ $# -eq 0 ]] || { echo "This wrapper takes no positional arguments." >&2; exit 2; }
[[ "$EVALUATION_SEED" =~ ^[0-9]+$ ]] || { echo "EVALUATION_SEED must be non-negative" >&2; exit 2; }
[[ "$TERRAIN_LEVEL" =~ ^[0-7]$ ]] || { echo "TERRAIN_LEVEL must be in [0,7]" >&2; exit 2; }
[[ "$VIDEO_LENGTH" =~ ^[1-9][0-9]*$ ]] || { echo "VIDEO_LENGTH must be positive" >&2; exit 2; }
[[ "$WARMUP_STEPS" =~ ^[0-9]+$ && "$WARMUP_STEPS" -lt "$VIDEO_LENGTH" ]] || {
    echo "WARMUP_STEPS must be non-negative and less than VIDEO_LENGTH" >&2; exit 2;
}
[[ "$DRY_RUN" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
[[ -x "$PYTHON_EXE" ]] || { echo "Python executable not found: $PYTHON_EXE" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 1; }
actual_sha256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
[[ "$actual_sha256" == "$EXPECTED_SHA256" ]] || {
    echo "Checkpoint SHA-256 mismatch" >&2; echo "Expected: $EXPECTED_SHA256" >&2
    echo "Actual:   $actual_sha256" >&2; exit 1;
}
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Refusing to overwrite: $OUTPUT_ROOT" >&2; exit 1; }

cd "$PROJECT_DIR"
echo "Task:                     $TASK"
echo "Task variant:             v5_boxes_safety_remediation_v2"
echo "Training seed:            $TRAINING_SEED"
echo "Evaluation scenario seed: $EVALUATION_SEED"
echo "Checkpoint:               $CHECKPOINT"
echo "Checkpoint SHA-256:       $actual_sha256"
echo "Terrain:                  boxes level $TERRAIN_LEVEL"
echo "Video steps:              $VIDEO_LENGTH"
echo "Output root:              $OUTPUT_ROOT"

if [[ "$DRY_RUN" == "0" ]]; then mkdir -p "$OUTPUT_ROOT"; fi
for label in "${LABELS[@]}"; do
    output_dir="${OUTPUT_ROOT}/${label}"
    console_log="${OUTPUT_ROOT}/${label}.console.log"
    command=("$PYTHON_EXE" scripts/project/record_go2_multiterrain_demo.py
        --task "$TASK" --checkpoint "$CHECKPOINT" --terrain_type boxes
        --terrain_level "$TERRAIN_LEVEL" --training_seed "$TRAINING_SEED" --seed "$EVALUATION_SEED"
        --video_length "$VIDEO_LENGTH" --warmup_steps "$WARMUP_STEPS"
        --command_x "${COMMAND_X[$label]}" --command_y "${COMMAND_Y[$label]}"
        --command_yaw "${COMMAND_YAW[$label]}" --camera_mode "$CAMERA_MODE"
        --output_dir "$output_dir" --device "$DEVICE")
    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'Dry-run command:'; printf ' %q' "${command[@]}"; printf '\n'; continue
    fi
    set +e
    "${command[@]}" 2>&1 | tee "$console_log"
    status=${PIPESTATUS[0]}
    set -e
    echo "Recording exit code ($label): $status"
    [[ "$status" -eq 0 ]] || exit "$status"
done
if [[ "$DRY_RUN" == "0" ]]; then
    find "$OUTPUT_ROOT" -type f -printf '%s %P\n' | sort -k2 > "$OUTPUT_ROOT/inventory.txt"
    find "$OUTPUT_ROOT" -type f ! -name final_artifacts.sha256 -print0 | sort -z | xargs -0 sha256sum > "$OUTPUT_ROOT/final_artifacts.sha256"
    echo "Recording matrix completed: $OUTPUT_ROOT"
fi
