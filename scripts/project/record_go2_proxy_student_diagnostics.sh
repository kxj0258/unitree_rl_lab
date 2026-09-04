#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
PYTHON_EXE="${PYTHON_EXE:-/home/kxj/miniforge3/envs/unitree_rl/bin/python}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_DIR}/logs/student_proxy_ablation/proxy_teacher_student_fromscratch_trainseed42_updates10000_v1/student_model_10000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/evaluation/video_proxy_student_model10000_trainseed42_diagnostics_v1}"
TASK="Unitree-Go2-Velocity-ProxyStudent-HeightScan"
TRAINING_SEED="${TRAINING_SEED:-42}"
EVALUATION_SEED="${EVALUATION_SEED:-42}"
VIDEO_LENGTH="${VIDEO_LENGTH:-500}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
DEVICE="${DEVICE:-cuda:0}"
ONLY_SCENARIO="${ONLY_SCENARIO:-all}"

[[ -d "$PROJECT_DIR" ]] || { echo "Project directory not found: $PROJECT_DIR" >&2; exit 1; }
[[ -x "$PYTHON_EXE" ]] || { echo "Python executable not found: $PYTHON_EXE" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 1; }
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Refusing to overwrite existing output: $OUTPUT_ROOT" >&2; exit 1; }

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT"

scenarios=(
    flat_level0_forward
    flat_level0_reverse
    flat_level0_lateral_right
    flat_level0_yaw_left
    inverted_pyramid_stairs_level7_forward
    boxes_level7_reverse
)

run_scenario() {
    local name="$1" terrain level command_x command_y command_yaw
    case "$name" in
        flat_level0_forward)
            terrain=flat; level=0; command_x=0.30; command_y=0.00; command_yaw=0.00 ;;
        flat_level0_reverse)
            terrain=flat; level=0; command_x=-0.30; command_y=0.00; command_yaw=0.00 ;;
        flat_level0_lateral_right)
            terrain=flat; level=0; command_x=0.00; command_y=-0.20; command_yaw=0.00 ;;
        flat_level0_yaw_left)
            terrain=flat; level=0; command_x=0.00; command_y=0.00; command_yaw=0.30 ;;
        inverted_pyramid_stairs_level7_forward)
            terrain=inverted_pyramid_stairs; level=7; command_x=0.30; command_y=0.00; command_yaw=0.00 ;;
        boxes_level7_reverse)
            terrain=boxes; level=7; command_x=-0.30; command_y=0.00; command_yaw=0.00 ;;
        *) echo "Unknown scenario: $name" >&2; return 2 ;;
    esac

    local output_dir="${OUTPUT_ROOT}/${name}_evalseed${EVALUATION_SEED}"
    local console_log="${OUTPUT_ROOT}/${name}_evalseed${EVALUATION_SEED}.console.log"
    "$PYTHON_EXE" scripts/project/record_go2_multiterrain_demo.py \
        --task "$TASK" \
        --checkpoint "$CHECKPOINT" \
        --terrain_type "$terrain" \
        --terrain_level "$level" \
        --training_seed "$TRAINING_SEED" \
        --seed "$EVALUATION_SEED" \
        --video_length "$VIDEO_LENGTH" \
        --warmup_steps "$WARMUP_STEPS" \
        --command_x "$command_x" \
        --command_y "$command_y" \
        --command_yaw "$command_yaw" \
        --camera_mode asset_follow \
        --output_dir "$output_dir" \
        --device "$DEVICE" \
        2>&1 | tee "$console_log"
    local status=${PIPESTATUS[0]}
    echo "$name exit code: $status"
    return "$status"
}

if [[ "$ONLY_SCENARIO" == all ]]; then
    selected=("${scenarios[@]}")
else
    selected=("$ONLY_SCENARIO")
fi

for scenario in "${selected[@]}"; do
    run_scenario "$scenario"
done

find "$OUTPUT_ROOT" -type f -printf '%s %P\n' | sort -k2 > "$OUTPUT_ROOT/inventory.txt"
find "$OUTPUT_ROOT" -type f ! -name 'all_diagnostic_evidence.sha256' -print0 \
    | sort -z | xargs -0 sha256sum > "$OUTPUT_ROOT/all_diagnostic_evidence.sha256"
sha256sum "$OUTPUT_ROOT/all_diagnostic_evidence.sha256"
echo "Proxy-student diagnostic recording finished: $OUTPUT_ROOT"
