#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/evaluation/video_rough_multitraining_model2999_evalseed42_2026-08-26}"
EVALUATION_SEED="${EVALUATION_SEED:-42}"
TERRAIN_LEVEL="${TERRAIN_LEVEL:-5}"
VIDEO_LENGTH="${VIDEO_LENGTH:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
COMMAND_X="${COMMAND_X:-0.20}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON_EXE="${PYTHON_EXE:-/home/kxj/miniforge3/envs/unitree_rl/bin/python}"

seeds=(42 7 123)
runs=(
  "2026-08-25_04-40-21_roughmix50_curriculum_formal_seed42_env2048_i3000"
  "2026-08-25_13-53-47_roughmix50_curriculum_formal_seed7_env2048_i3000"
  "2026-08-25_17-23-39_roughmix50_curriculum_formal_seed123_env2048_i3000"
)

if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite existing output: $OUTPUT_ROOT" >&2
    exit 1
fi
if [[ ! -x "$PYTHON_EXE" ]]; then
    echo "Python executable not found: $PYTHON_EXE" >&2
    exit 1
fi
mkdir -p "$OUTPUT_ROOT"

for index in "${!seeds[@]}"; do
    training_seed="${seeds[$index]}"
    checkpoint="${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity_rough/${runs[$index]}/model_2999.pt"
    output_dir="${OUTPUT_ROOT}/trainseed${training_seed}_model2999_random_rough_level${TERRAIN_LEVEL}"
    console_log="${OUTPUT_ROOT}/trainseed${training_seed}_model2999_random_rough_level${TERRAIN_LEVEL}.console.log"
    if [[ ! -f "$checkpoint" ]]; then
        echo "Checkpoint not found: $checkpoint" >&2
        exit 1
    fi
    "$PYTHON_EXE" scripts/project/record_go2_multiterrain_demo.py \
        --task Unitree-Go2-Velocity-Rough \
        --checkpoint "$checkpoint" \
        --terrain_type random_rough \
        --terrain_level "$TERRAIN_LEVEL" \
        --training_seed "$training_seed" \
        --seed "$EVALUATION_SEED" \
        --video_length "$VIDEO_LENGTH" \
        --warmup_steps "$WARMUP_STEPS" \
        --command_x "$COMMAND_X" \
        --output_dir "$output_dir" \
        --device "$DEVICE" \
        2>&1 | tee "$console_log"
    status=${PIPESTATUS[0]}
    if [[ "$status" -ne 0 ]]; then
        echo "Recording failed for training seed $training_seed with exit code $status" >&2
        exit "$status"
    fi
done

find "$OUTPUT_ROOT" -type f -printf '%s %P\n' | sort -k2 > "$OUTPUT_ROOT/inventory.txt"
find "$OUTPUT_ROOT" -type f ! -name 'final_artifacts.sha256' -print0 | sort -z | xargs -0 sha256sum > "$OUTPUT_ROOT/final_artifacts.sha256"
echo "[ROUGH_DEMO_MATRIX] completed successfully: $OUTPUT_ROOT"
