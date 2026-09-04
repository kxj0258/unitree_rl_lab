#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
PYTHON_BIN="${PYTHON_BIN:-/home/kxj/miniforge3/envs/unitree_rl/bin/python}"
NUM_ENVS="${NUM_ENVS:-20}"
EPISODES_PER_ENV="${EPISODES_PER_ENV:-2}"
DEVICE="${DEVICE:-cuda:0}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/evaluation/teacher_v1_finalists_trainseed42_crossseed_random_commands_v1}"

[[ -d "$PROJECT_DIR" ]] || { echo "Project directory not found: $PROJECT_DIR" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 1; }
[[ "$NUM_ENVS" =~ ^[0-9]+$ ]] && (( NUM_ENVS >= 10 && NUM_ENVS % 10 == 0 )) || {
    echo "NUM_ENVS must be a positive multiple of 10" >&2
    exit 1
}
[[ "$EPISODES_PER_ENV" =~ ^[1-9][0-9]*$ ]] || { echo "EPISODES_PER_ENV must be positive" >&2; exit 1; }

declare -A TASK_SUFFIX=(
    [flat]="Flat"
    [random_rough]="RandomRough"
    [pyramid_slope]="PyramidSlope"
    [inverted_pyramid_slope]="InvertedPyramidSlope"
    [pyramid_stairs]="PyramidStairs"
    [inverted_pyramid_stairs]="InvertedPyramidStairs"
    [boxes]="Boxes"
)

declare -A RUN_DIR=(
    [flat]="logs/rsl_rl/unitree_go2_velocity_teacher_v1_flat/2026-08-29_16-48-29_teacher_v1_flat_fromscratch_trainseed42_env1024_i5000"
    [random_rough]="logs/rsl_rl/unitree_go2_velocity_teacher_v1_random_rough/2026-08-29_19-35-33_teacher_v1_random_rough_fromscratch_trainseed42_env1024_i5000"
    [pyramid_slope]="logs/rsl_rl/unitree_go2_velocity_teacher_v1_pyramid_slope/2026-08-29_22-31-02_teacher_v1_pyramid_slope_fromscratch_trainseed42_env1024_i5000"
    [inverted_pyramid_slope]="logs/rsl_rl/unitree_go2_velocity_teacher_v1_inverted_pyramid_slope/2026-08-30_01-28-47_teacher_v1_inverted_pyramid_slope_fromscratch_trainseed42_env1024_i5000"
    [pyramid_stairs]="logs/rsl_rl/unitree_go2_velocity_teacher_v1_pyramid_stairs/2026-08-30_04-15-42_teacher_v1_pyramid_stairs_fromscratch_trainseed42_env1024_i5000"
    [inverted_pyramid_stairs]="logs/rsl_rl/unitree_go2_velocity_teacher_v1_inverted_pyramid_stairs/2026-08-30_07-13-57_teacher_v1_inverted_pyramid_stairs_fromscratch_trainseed42_env1024_i5000"
    [boxes]="logs/rsl_rl/unitree_go2_velocity_teacher_v1_boxes/2026-08-30_10-23-44_teacher_v1_boxes_fromscratch_trainseed42_env1024_i5000"
)

# Each entry is terrain:checkpoint.  Ambiguous terrains intentionally retain
# two checkpoints until cross-seed evidence resolves the choice.
finalists=(
    flat:4999
    random_rough:2000
    random_rough:4999
    pyramid_slope:4000
    inverted_pyramid_slope:4000
    pyramid_stairs:4700
    inverted_pyramid_stairs:4300
    inverted_pyramid_stairs:4999
    boxes:4999
)
evaluation_seeds=(7 123)

cd "$PROJECT_DIR"
if [[ "$DRY_RUN" != 1 && -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to reuse existing output root: $OUTPUT_ROOT" >&2
    exit 1
fi

for entry in "${finalists[@]}"; do
    terrain="${entry%%:*}"
    iteration="${entry##*:}"
    checkpoint="$PROJECT_DIR/${RUN_DIR[$terrain]}/model_${iteration}.pt"
    [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }

    for evaluation_seed in "${evaluation_seeds[@]}"; do
        output_dir="$OUTPUT_ROOT/${terrain}_model${iteration}_trainseed42_evalseed${evaluation_seed}"
        cmd=(
            "$PYTHON_BIN" scripts/project/evaluate_go2.py
            --task "Unitree-Go2-Velocity-TeacherV1-${TASK_SUFFIX[$terrain]}"
            --terrain_protocol teacher_v1_stratified
            --num_envs "$NUM_ENVS"
            --episodes_per_env "$EPISODES_PER_ENV"
            --seed "$evaluation_seed"
            --checkpoint "$checkpoint"
            --output_dir "$output_dir"
            --device "$DEVICE"
            --headless
        )
        printf 'COMMAND:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        if [[ "$DRY_RUN" != 1 ]]; then
            mkdir -p "$OUTPUT_ROOT/console"
            "${cmd[@]}" 2>&1 | tee "$OUTPUT_ROOT/console/${terrain}_model${iteration}_trainseed42_evalseed${evaluation_seed}.console.log"
            echo "$terrain model$iteration evalseed=$evaluation_seed exit code: 0"
        fi
    done
done

if [[ "$DRY_RUN" == 1 ]]; then
    echo "Terrain Teacher v1 cross-seed finalist dry-run completed; no evaluation launched."
    exit 0
fi

find "$OUTPUT_ROOT" -type f ! -name all_crossseed_finalist_evidence.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$OUTPUT_ROOT/all_crossseed_finalist_evidence.sha256"
sha256sum "$OUTPUT_ROOT/all_crossseed_finalist_evidence.sha256"
echo "Terrain Teacher v1 cross-seed finalist evaluation finished: $OUTPUT_ROOT"
