#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
PYTHON_EXE="${PYTHON_EXE:-/home/kxj/miniforge3/envs/unitree_rl/bin/python}"
EVALUATION_SEED="${EVALUATION_SEED:-42}"
TRAINING_SEED="${TRAINING_SEED:-42}"
COMMAND_X="${COMMAND_X:-0.40}"
VIDEO_LENGTH="${VIDEO_LENGTH:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
DEVICE="${DEVICE:-cuda:0}"
DATE_TAG="${DATE_TAG:-2026-09-02}"

pyramid_checkpoint="${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity_teacher_v2_pyramid_stairs_remediation_v2/2026-09-02_00-44-15_teacher_v2_pyramid_stairs_remediation_v2_gaitclock_participation_fromscratch_formal_trainseed42_env1024_i5000/model_4999.pt"
inverted_checkpoint="${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity_teacher_v2_inverted_pyramid_stairs_remediation/2026-09-02_03-42-09_teacher_v2_inverted_pyramid_stairs_remediation_ascent_heightscan_gaitclock_participation_anticircle_fromscratch_formal_trainseed42_env1024_i5000/model_4999.pt"
boxes_checkpoint="${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity_teacher_v2_boxes_remediation/2026-09-02_06-40-36_teacher_v2_boxes_remediation_edgeaware_heightscan_gaitclock_participation_anticircle_fromscratch_formal_trainseed42_env1024_i5000/model_4999.pt"

record_case() {
    local label="$1" task="$2" terrain="$3" level="$4" checkpoint="$5" output_dir="$6"
    local console_log="${output_dir}.console.log"
    [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; return 2; }
    [[ ! -e "$output_dir" && ! -e "$console_log" ]] || {
        echo "Refusing to overwrite gait evidence: $output_dir" >&2
        return 2
    }
    echo "================================================================"
    echo "Task type:                Teacher v2 gait gate recording"
    echo "Label:                    $label"
    echo "Task:                     $task"
    echo "Terrain:                  $terrain"
    echo "Terrain level:            $level"
    echo "Training seed:            $TRAINING_SEED"
    echo "Evaluation scenario seed: $EVALUATION_SEED"
    echo "Source checkpoint:        $checkpoint"
    echo "Output directory:         $output_dir"
    echo "Start time:               $(date --iso-8601=seconds)"
    echo "================================================================"
    set +e
    "$PYTHON_EXE" scripts/project/record_go2_multiterrain_demo.py \
        --task "$task" \
        --checkpoint "$checkpoint" \
        --terrain_type "$terrain" \
        --terrain_level "$level" \
        --training_seed "$TRAINING_SEED" \
        --seed "$EVALUATION_SEED" \
        --video_length "$VIDEO_LENGTH" \
        --warmup_steps "$WARMUP_STEPS" \
        --command_x "$COMMAND_X" \
        --command_y 0.0 \
        --fixed_relative_heading 0.0 \
        --camera_mode terrain_fixed \
        --output_dir "$output_dir" \
        --device "$DEVICE" \
        --headless 2>&1 | tee "$console_log"
    local status=${PIPESTATUS[0]}
    set -e
    echo "$label recording exit code: $status"
    [[ "$status" -eq 0 ]] || return "$status"

    "$PYTHON_EXE" scripts/project/analyze_go2_gait_contacts.py \
        "$output_dir/tracking_timeseries.csv" \
        --warmup_steps "$WARMUP_STEPS" \
        --output_json "$output_dir/gait_analysis.json"
    status=$?
    echo "$label gait analysis exit code: $status"
    [[ "$status" -eq 0 ]] || return "$status"
    sha256sum "$output_dir/gait_analysis.json" "$output_dir/metadata.json" \
        "$output_dir/tracking_timeseries.csv" "$output_dir"/video/*.mp4 "$console_log"
}

cd "$PROJECT_DIR"

record_case \
    "Pyramid Stairs remediation-v2 model4999 level5" \
    "Unitree-Go2-Velocity-TeacherV2-PyramidStairsRemediationV2" \
    "pyramid_stairs" 5 "$pyramid_checkpoint" \
    "${PROJECT_DIR}/evaluation/teacher_v2_pyramid_stairs_remediation_v2_gait_gate_model4999_trainseed42_evalseed${EVALUATION_SEED}_level5_vx040_heading0_20s_${DATE_TAG}"

record_case \
    "Inverted Pyramid Stairs remediation model4999 level5" \
    "Unitree-Go2-Velocity-TeacherV2-InvertedPyramidStairsRemediation" \
    "inverted_pyramid_stairs" 5 "$inverted_checkpoint" \
    "${PROJECT_DIR}/evaluation/teacher_v2_inverted_pyramid_stairs_remediation_gait_gate_model4999_trainseed42_evalseed${EVALUATION_SEED}_level5_vx040_heading0_20s_${DATE_TAG}"

for level in 1 3 5; do
    record_case \
        "Boxes remediation model4999 level${level}" \
        "Unitree-Go2-Velocity-TeacherV2-BoxesRemediation" \
        "boxes" "$level" "$boxes_checkpoint" \
        "${PROJECT_DIR}/evaluation/teacher_v2_boxes_remediation_gait_gate_model4999_trainseed42_evalseed${EVALUATION_SEED}_level${level}_vx040_heading0_20s_${DATE_TAG}"
done

echo "All remaining Teacher v2 gait-gate recordings and analyses completed successfully."
