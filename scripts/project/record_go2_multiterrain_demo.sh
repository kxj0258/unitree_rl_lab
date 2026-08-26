#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity_multiterrain/2026-08-25_22-53-12_multiterrain_v3_challenging_noheightscan_prototype_resume_from2396_seed42_env1024_addi600_totalapprox3000/model_2995.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/evaluation/video_multiterrain_stratified_v0_model2995_evalseed42_2026-08-26}"
EVALUATION_SEED="${EVALUATION_SEED:-42}"
TRAINING_SEED="${TRAINING_SEED:-42}"
TERRAIN_LEVEL="${TERRAIN_LEVEL:-5}"
VIDEO_LENGTH="${VIDEO_LENGTH:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
COMMAND_X="${COMMAND_X:-0.20}"
CAMERA_MODE="${CAMERA_MODE:-terrain_fixed}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON_EXE="${PYTHON_EXE:-/home/kxj/miniforge3/envs/unitree_rl/bin/python}"

terrains=(flat random_rough pyramid_slope inverted_pyramid_slope pyramid_stairs inverted_pyramid_stairs boxes)

if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite existing output: $OUTPUT_ROOT" >&2
    exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi
if [[ ! -x "$PYTHON_EXE" ]]; then
    echo "Python executable not found: $PYTHON_EXE" >&2
    exit 1
fi
mkdir -p "$OUTPUT_ROOT"

for terrain in "${terrains[@]}"; do
    output_dir="${OUTPUT_ROOT}/${terrain}_level${TERRAIN_LEVEL}"
    console_log="${OUTPUT_ROOT}/${terrain}_level${TERRAIN_LEVEL}.console.log"
    "$PYTHON_EXE" scripts/project/record_go2_multiterrain_demo.py \
        --task Unitree-Go2-Velocity-MultiTerrain \
        --checkpoint "$CHECKPOINT" \
        --terrain_type "$terrain" \
        --terrain_level "$TERRAIN_LEVEL" \
        --training_seed "$TRAINING_SEED" \
        --seed "$EVALUATION_SEED" \
        --video_length "$VIDEO_LENGTH" \
        --warmup_steps "$WARMUP_STEPS" \
        --command_x "$COMMAND_X" \
        --camera_mode "$CAMERA_MODE" \
        --output_dir "$output_dir" \
        --device "$DEVICE" \
        2>&1 | tee "$console_log"
    status=${PIPESTATUS[0]}
    if [[ "$status" -ne 0 ]]; then
        echo "Recording failed for $terrain with exit code $status" >&2
        exit "$status"
    fi
done

find "$OUTPUT_ROOT" -type f -printf '%s %P\n' | sort -k2 > "$OUTPUT_ROOT/inventory.txt"
find "$OUTPUT_ROOT" -type f ! -name 'final_artifacts.sha256' -print0 | sort -z | xargs -0 sha256sum > "$OUTPUT_ROOT/final_artifacts.sha256"
echo "[DEMO_MATRIX] completed successfully: $OUTPUT_ROOT"
