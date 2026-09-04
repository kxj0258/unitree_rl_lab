#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
PYTHON_EXE="${PYTHON_EXE:-/home/kxj/miniforge3/envs/unitree_rl/bin/python}"
CHECKPOINT="${CHECKPOINT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
TASK="${TASK:-Unitree-Go2-Velocity-MultiTerrain}"
TRAINING_SEED="${TRAINING_SEED:-42}"
EVALUATION_SEED="${EVALUATION_SEED:-42}"
TERRAIN_LEVEL="${TERRAIN_LEVEL:-5}"
COMMAND_SPEED="${COMMAND_SPEED:-0.20}"
VIDEO_LENGTH="${VIDEO_LENGTH:-600}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
CAMERA_MODE="${CAMERA_MODE:-asset_follow}"
DEVICE="${DEVICE:-cuda:0}"
INCLUDE_BOXES_REVERSE="${INCLUDE_BOXES_REVERSE:-0}"

if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    echo "CHECKPOINT must name an existing checkpoint" >&2
    exit 2
fi
if [[ -z "$OUTPUT_ROOT" ]]; then
    echo "OUTPUT_ROOT is required" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite existing output: $OUTPUT_ROOT" >&2
    exit 1
fi
if [[ ! -x "$PYTHON_EXE" ]]; then
    echo "Python executable not found: $PYTHON_EXE" >&2
    exit 1
fi
if [[ "$INCLUDE_BOXES_REVERSE" != "0" && "$INCLUDE_BOXES_REVERSE" != "1" ]]; then
    echo "INCLUDE_BOXES_REVERSE must be 0 or 1" >&2
    exit 2
fi
if [[ "$TASK" != "Unitree-Go2-Velocity-MultiTerrain" && \
      "$TASK" != "Unitree-Go2-Velocity-MultiTerrain-HeightScan" ]]; then
    echo "Unsupported TASK for directional diagnostics: $TASK" >&2
    exit 2
fi

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT"

terrains=(pyramid_slope inverted_pyramid_slope pyramid_stairs inverted_pyramid_stairs boxes)
for terrain in "${terrains[@]}"; do
    directions=(positive)
    if [[ "$terrain" != "boxes" || "$INCLUDE_BOXES_REVERSE" == "1" ]]; then
        directions+=(negative)
    fi
    for direction in "${directions[@]}"; do
        command_x="${COMMAND_SPEED#-}"
        if [[ "$direction" == "negative" ]]; then
            command_x="-${COMMAND_SPEED#-}"
        fi
        output_dir="$OUTPUT_ROOT/${terrain}_level${TERRAIN_LEVEL}_${direction}_vx${COMMAND_SPEED#-}"
        console_log="$OUTPUT_ROOT/${terrain}_level${TERRAIN_LEVEL}_${direction}.console.log"
        "$PYTHON_EXE" scripts/project/record_go2_multiterrain_demo.py \
            --task "$TASK" \
            --checkpoint "$CHECKPOINT" \
            --terrain_type "$terrain" \
            --terrain_level "$TERRAIN_LEVEL" \
            --training_seed "$TRAINING_SEED" \
            --seed "$EVALUATION_SEED" \
            --video_length "$VIDEO_LENGTH" \
            --warmup_steps "$WARMUP_STEPS" \
            --command_x "$command_x" \
            --camera_mode "$CAMERA_MODE" \
            --output_dir "$output_dir" \
            --device "$DEVICE" \
            2>&1 | tee "$console_log"
    done
done

"$PYTHON_EXE" scripts/project/analyze_go2_multiterrain_diagnostics.py \
    --input_root "$OUTPUT_ROOT"
find "$OUTPUT_ROOT" -type f -printf '%s %P\n' | sort -k2 > "$OUTPUT_ROOT/inventory.txt"
find "$OUTPUT_ROOT" -type f ! -name 'final_artifacts.sha256' -print0 \
    | sort -z | xargs -0 sha256sum > "$OUTPUT_ROOT/final_artifacts.sha256"
echo "[DIAGNOSTIC] completed successfully: $OUTPUT_ROOT"
