#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
PYTHON_EXE="${PYTHON_EXE:-/home/kxj/miniforge3/envs/unitree_rl/bin/python}"
CHECKPOINT="${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity_v5_stairs_specialist/2026-08-27_16-17-34_v5_perceptive_gru_stairs_specialist_formal_continue_from2200_trainseed42_env1024_addi2800_total5000/model_4999.pt"
EXPECTED_SHA256="47b42309c03450c88f27ceaf66ed393abfcc5a488e7a508a369055ed4cfcee50"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/evaluation/video_v5_stairs_directional_diagnostic_model4999_trainseed42_evalseed42_level6_vx035_20s_2026-08-27}"
TRAINING_SEED=42
EVALUATION_SEED="${EVALUATION_SEED:-42}"
TERRAIN_LEVEL="${TERRAIN_LEVEL:-6}"
COMMAND_X="${COMMAND_X:-0.35}"
VIDEO_LENGTH="${VIDEO_LENGTH:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
CAMERA_MODE="${CAMERA_MODE:-asset_follow}"
DEVICE="${DEVICE:-cuda:0}"
DRY_RUN="${DRY_RUN:-0}"
TERRAINS=(pyramid_stairs inverted_pyramid_stairs)
COMMAND_TAG="${COMMAND_X//./}"

if [[ $# -ne 0 ]]; then
    echo "This wrapper takes no positional arguments." >&2
    exit 2
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1: $DRY_RUN" >&2
    exit 2
fi
if [[ ! "$TERRAIN_LEVEL" =~ ^[0-7]$ ]]; then
    echo "TERRAIN_LEVEL must be in [0, 7]: $TERRAIN_LEVEL" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_EXE" ]]; then
    echo "Python executable not found: $PYTHON_EXE" >&2
    exit 1
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
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite existing output: $OUTPUT_ROOT" >&2
    exit 1
fi

echo "Task:                     Unitree-Go2-Velocity-V5-Stairs-Specialist"
echo "Task variant:             v5_stairs_specialist_directional_video_diagnostic"
echo "Training seed:            $TRAINING_SEED"
echo "Evaluation scenario seed: $EVALUATION_SEED"
echo "Checkpoint:               $CHECKPOINT"
echo "Checkpoint SHA-256:       $actual_sha256"
echo "Terrain level:            $TERRAIN_LEVEL"
echo "Fixed command:            vx=$COMMAND_X vy=0 yaw=0"
echo "Video steps:              $VIDEO_LENGTH"
echo "Warmup steps:             $WARMUP_STEPS"
echo "Camera mode:              $CAMERA_MODE"
echo "Output root:              $OUTPUT_ROOT"

if [[ "$DRY_RUN" == "0" ]]; then
    mkdir -p "$OUTPUT_ROOT"
fi
cd "$PROJECT_DIR"

for terrain in "${TERRAINS[@]}"; do
    output_dir="$OUTPUT_ROOT/${terrain}_level${TERRAIN_LEVEL}_vx${COMMAND_TAG}"
    console_log="$OUTPUT_ROOT/${terrain}_level${TERRAIN_LEVEL}_vx${COMMAND_TAG}.console.log"
    command=(
        "$PYTHON_EXE" scripts/project/record_go2_multiterrain_demo.py
        --task Unitree-Go2-Velocity-V5-Stairs-Specialist
        --checkpoint "$CHECKPOINT"
        --terrain_type "$terrain"
        --terrain_level "$TERRAIN_LEVEL"
        --training_seed "$TRAINING_SEED"
        --seed "$EVALUATION_SEED"
        --video_length "$VIDEO_LENGTH"
        --warmup_steps "$WARMUP_STEPS"
        --command_x "$COMMAND_X"
        --command_y 0.0
        --command_yaw 0.0
        --camera_mode "$CAMERA_MODE"
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
    status=${PIPESTATUS[0]}
    set -e
    echo "Recording exit code ($terrain): $status"
    if [[ "$status" -ne 0 ]]; then
        exit "$status"
    fi
done

if [[ "$DRY_RUN" == "0" ]]; then
    find "$OUTPUT_ROOT" -type f -printf '%s %P\n' | sort -k2 > "$OUTPUT_ROOT/inventory.txt"
    find "$OUTPUT_ROOT" -type f ! -name 'final_artifacts.sha256' -print0 \
        | sort -z | xargs -0 sha256sum > "$OUTPUT_ROOT/final_artifacts.sha256"
    echo "[V5_STAIRS_DIRECTIONAL_DIAGNOSTIC] completed successfully: $OUTPUT_ROOT"
fi
