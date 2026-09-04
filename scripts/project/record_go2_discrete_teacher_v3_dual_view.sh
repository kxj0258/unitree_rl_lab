#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
PYTHON_EXE="${PYTHON_EXE:-/home/kxj/miniforge3/envs/unitree_rl/bin/python}"
TRAINING_SEED="${TRAINING_SEED:-42}"
EVALUATION_SEED="${EVALUATION_SEED:-42}"
TERRAIN_LEVEL="${TERRAIN_LEVEL:-5}"
COMMAND_X="${COMMAND_X:-0.40}"
VIDEO_LENGTH="${VIDEO_LENGTH:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
DATE_TAG="${DATE_TAG:-2026-09-03}"
DEVICE="${DEVICE:-cuda:0}"
MANIFEST="${PROJECT_DIR}/evaluation/teacher_v3_dual_view_gait_gate_trainseed${TRAINING_SEED}_evalseed${EVALUATION_SEED}_${DATE_TAG}.sha256"

[[ "$TRAINING_SEED" == "42" ]] || { echo "This evidence script is pinned to training seed 42" >&2; exit 2; }
[[ ! -e "$MANIFEST" ]] || { echo "Refusing to overwrite V3 gait-gate manifest: $MANIFEST" >&2; exit 2; }

tasks=(
    "Unitree-Go2-Velocity-TeacherV3-PyramidStairs"
    "Unitree-Go2-Velocity-TeacherV3-InvertedPyramidStairs"
    "Unitree-Go2-Velocity-TeacherV3-Boxes"
)
terrains=("pyramid_stairs" "inverted_pyramid_stairs" "boxes")
experiments=(
    "unitree_go2_velocity_teacher_v3_pyramid_stairs"
    "unitree_go2_velocity_teacher_v3_inverted_pyramid_stairs"
    "unitree_go2_velocity_teacher_v3_boxes"
)
run_dirs=(
    "2026-09-02_10-57-34_teacher_v3_pyramid_stairs_worstleg_symmetry_heightscan_anticircle_fromscratch_formal_trainseed42_env2048_i5000_env2048_i5000_budget327m_fromscratch_v1"
    "2026-09-02_16-54-00_teacher_v3_inverted_pyramid_stairs_worstleg_symmetry_heightscan_ascent_anticircle_fromscratch_formal_trainseed42_env2048_i5000_env2048_i5000_budget327m_fromscratch_v1"
    "2026-09-02_21-52-51_teacher_v3_boxes_worstleg_symmetry_edgeaware_mixedlevel_heightscan_anticircle_fromscratch_formal_trainseed42_env2048_i5000_env2048_i5000_budget327m_fromscratch_v1"
)
iterations=(4999 4700 4999)

cd "$PROJECT_DIR"
evidence_files=()
for index in 0 1 2; do
    experiment_root="${PROJECT_DIR}/logs/rsl_rl/${experiments[$index]}"
    run_dir="${experiment_root}/${run_dirs[$index]}"
    iteration="${iterations[$index]}"
    checkpoint="${run_dir}/model_${iteration}.pt"
    [[ -f "$checkpoint" ]] || { echo "Missing V3 checkpoint: $checkpoint" >&2; exit 2; }
    for view in terrain_fixed asset_follow; do
        output_dir="${PROJECT_DIR}/evaluation/teacher_v3_${terrains[$index]}_gait_gate_model${iteration}_trainseed${TRAINING_SEED}_evalseed${EVALUATION_SEED}_level${TERRAIN_LEVEL}_vx040_heading0_20s_${view}_${DATE_TAG}"
        console_log="${output_dir}.console.log"
        [[ ! -e "$output_dir" && ! -e "$console_log" ]] || {
            echo "Refusing to overwrite V3 dual-view evidence: $output_dir" >&2
            exit 2
        }
        echo "================================================================"
        echo "Task type:                Teacher V3 dual-view gait gate"
        echo "Task:                     ${tasks[$index]}"
        echo "Terrain:                  ${terrains[$index]}"
        echo "Variant:                  V3 remediation"
        echo "Training seed:            $TRAINING_SEED"
        echo "Evaluation scenario seed: $EVALUATION_SEED"
        echo "Source/current checkpoint:$checkpoint"
        echo "Current iteration:        $iteration"
        echo "Terrain level:            $TERRAIN_LEVEL"
        echo "Command:                  vx=$COMMAND_X, vy=0, relative heading=0"
        echo "Camera view:              $view"
        echo "Output directory:         $output_dir"
        echo "Start time:               $(date --iso-8601=seconds)"
        echo "================================================================"
        set +e
        "$PYTHON_EXE" scripts/project/record_go2_multiterrain_demo.py \
            --task "${tasks[$index]}" --checkpoint "$checkpoint" \
            --terrain_type "${terrains[$index]}" --terrain_level "$TERRAIN_LEVEL" \
            --training_seed "$TRAINING_SEED" --seed "$EVALUATION_SEED" \
            --video_length "$VIDEO_LENGTH" --warmup_steps "$WARMUP_STEPS" \
            --command_x "$COMMAND_X" --command_y 0.0 --fixed_relative_heading 0.0 \
            --camera_mode "$view" --output_dir "$output_dir" --device "$DEVICE" --headless \
            2>&1 | tee "$console_log"
        status=${PIPESTATUS[0]}
        set -e
        echo "${terrains[$index]} $view recording exit code: $status"
        echo "End time: $(date --iso-8601=seconds)"
        [[ "$status" -eq 0 ]] || exit "$status"
        "$PYTHON_EXE" scripts/project/analyze_go2_gait_contacts.py \
            "$output_dir/tracking_timeseries.csv" --warmup_steps "$WARMUP_STEPS" \
            --output_json "$output_dir/gait_analysis.json"
        sha256sum "$output_dir/gait_analysis.json" "$output_dir/metadata.json" \
            "$output_dir/tracking_timeseries.csv" "$output_dir"/video/*.mp4 "$console_log"
        evidence_files+=(
            "$output_dir/gait_analysis.json"
            "$output_dir/metadata.json"
            "$output_dir/tracking_timeseries.csv"
            "$output_dir"/video/*.mp4
            "$console_log"
        )
    done
done

printf '%s\0' "${evidence_files[@]}" | sort -z | xargs -0 sha256sum >"$MANIFEST"
sha256sum "$MANIFEST"
echo "All Teacher V3 terrain-fixed and asset-follow recordings completed successfully."
echo "Evidence manifest: $MANIFEST"
