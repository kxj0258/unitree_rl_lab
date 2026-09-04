#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
PYTHON_BIN="${PYTHON_BIN:-/home/kxj/miniforge3/envs/unitree_rl/bin/python}"
EVALUATION_SEED="${EVALUATION_SEED:-123}"
NUM_ENVS="${NUM_ENVS:-100}"
EPISODES_PER_ENV="${EPISODES_PER_ENV:-2}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/evaluation/teacher_v3_candidate_screen_trainseed42_evalseed${EVALUATION_SEED}_randomcommand_2026-09-03}"

[[ -d "$PROJECT_DIR" ]] || { echo "Project directory not found: $PROJECT_DIR" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 1; }
[[ "$EVALUATION_SEED" =~ ^[0-9]+$ ]] || { echo "EVALUATION_SEED must be non-negative" >&2; exit 2; }
[[ "$NUM_ENVS" =~ ^[1-9][0-9]*$ ]] && (( NUM_ENVS >= 10 && NUM_ENVS % 10 == 0 )) || {
    echo "NUM_ENVS must be a positive multiple of 10" >&2
    exit 2
}
[[ "$EPISODES_PER_ENV" =~ ^[1-9][0-9]*$ ]] || { echo "EPISODES_PER_ENV must be positive" >&2; exit 2; }
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Refusing to reuse output root: $OUTPUT_ROOT" >&2; exit 1; }

declare -A TASK=(
    [pyramid_stairs]="Unitree-Go2-Velocity-TeacherV3-PyramidStairs"
    [inverted_pyramid_stairs]="Unitree-Go2-Velocity-TeacherV3-InvertedPyramidStairs"
    [boxes]="Unitree-Go2-Velocity-TeacherV3-Boxes"
)
declare -A RUN_DIR=(
    [pyramid_stairs]="logs/rsl_rl/unitree_go2_velocity_teacher_v3_pyramid_stairs/2026-09-02_10-57-34_teacher_v3_pyramid_stairs_worstleg_symmetry_heightscan_anticircle_fromscratch_formal_trainseed42_env2048_i5000_env2048_i5000_budget327m_fromscratch_v1"
    [inverted_pyramid_stairs]="logs/rsl_rl/unitree_go2_velocity_teacher_v3_inverted_pyramid_stairs/2026-09-02_16-54-00_teacher_v3_inverted_pyramid_stairs_worstleg_symmetry_heightscan_ascent_anticircle_fromscratch_formal_trainseed42_env2048_i5000_env2048_i5000_budget327m_fromscratch_v1"
    [boxes]="logs/rsl_rl/unitree_go2_velocity_teacher_v3_boxes/2026-09-02_21-52-51_teacher_v3_boxes_worstleg_symmetry_edgeaware_mixedlevel_heightscan_anticircle_fromscratch_formal_trainseed42_env2048_i5000_env2048_i5000_budget327m_fromscratch_v1"
)

checkpoints_for() {
    case "$1" in
        pyramid_stairs) echo "4700 4800 4999" ;;
        inverted_pyramid_stairs) echo "4700 4900 4999" ;;
        boxes) echo "4500 4900 4999" ;;
        *) return 1 ;;
    esac
}

mkdir -p "$OUTPUT_ROOT/console"
cd "$PROJECT_DIR"

for terrain in pyramid_stairs inverted_pyramid_stairs boxes; do
    for iteration in $(checkpoints_for "$terrain"); do
        checkpoint="${PROJECT_DIR}/${RUN_DIR[$terrain]}/model_${iteration}.pt"
        output_dir="${OUTPUT_ROOT}/${terrain}_model${iteration}_trainseed42_evalseed${EVALUATION_SEED}_randomcommand"
        console_log="${OUTPUT_ROOT}/console/${terrain}_model${iteration}_trainseed42_evalseed${EVALUATION_SEED}_randomcommand.console.log"
        [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
        [[ ! -e "$output_dir" && ! -e "$console_log" ]] || {
            echo "Refusing to overwrite evaluation evidence for $terrain model$iteration" >&2
            exit 1
        }

        echo "================================================================"
        echo "Task type:                Teacher V3 random-command candidate screen"
        echo "Task:                     ${TASK[$terrain]}"
        echo "Terrain:                  $terrain"
        echo "Variant:                  V3 remediation"
        echo "Training run:             ${RUN_DIR[$terrain]}"
        echo "Training seed:            42"
        echo "Evaluation scenario seed: $EVALUATION_SEED"
        echo "Source checkpoint:        $checkpoint"
        echo "Current checkpoint:       model_${iteration}.pt"
        echo "Output directory:         $output_dir"
        echo "Start time:               $(date --iso-8601=seconds)"
        echo "================================================================"

        set +e
        "$PYTHON_BIN" scripts/project/evaluate_go2.py \
            --task "${TASK[$terrain]}" \
            --terrain_protocol teacher_v3_stratified \
            --num_envs "$NUM_ENVS" \
            --episodes_per_env "$EPISODES_PER_ENV" \
            --seed "$EVALUATION_SEED" \
            --checkpoint "$checkpoint" \
            --output_dir "$output_dir" \
            --device "$DEVICE" \
            --headless 2>&1 | tee "$console_log"
        status=${PIPESTATUS[0]}
        set -e

        echo "End time:             $(date --iso-8601=seconds)"
        echo "Evaluation exit code: $status"
        [[ "$status" -eq 0 ]] || exit "$status"
        sha256sum "$output_dir/metadata.json" "$output_dir/episodes.csv" "$output_dir/summary.json" "$console_log"
    done
done

find "$OUTPUT_ROOT" -type f ! -name all_candidate_screen_evidence.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum >"$OUTPUT_ROOT/all_candidate_screen_evidence.sha256"
sha256sum "$OUTPUT_ROOT/all_candidate_screen_evidence.sha256"
echo "All Teacher V3 candidate-screen evaluations completed successfully."
echo "Output root: $OUTPUT_ROOT"
