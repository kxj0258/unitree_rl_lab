#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
PYTHON_BIN="${PYTHON_BIN:-/home/kxj/miniforge3/envs/unitree_rl/bin/python}"
NUM_ENVS="${NUM_ENVS:-20}"
EPISODES_PER_ENV="${EPISODES_PER_ENV:-2}"
DEVICE="${DEVICE:-cuda:0}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/evaluation/teacher_v1_finalists_trainseed42_fixed_command_matrix_seeds7_42_123_v1}"

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
declare -A ITERATION=(
    [flat]=4999
    [random_rough]=4999
    [pyramid_slope]=4000
    [inverted_pyramid_slope]=4000
    [pyramid_stairs]=4700
    [inverted_pyramid_stairs]=4999
    [boxes]=4999
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

terrains=(flat random_rough pyramid_slope inverted_pyramid_slope pyramid_stairs inverted_pyramid_stairs boxes)
evaluation_seeds=(7 42 123)

# Output records are: label vx vy yaw.  Every point lies inside that terrain's
# training command domain.  Discrete terrains always retain forward vx >= 0.15.
commands_for() {
    case "$1" in
        flat)
            printf '%s\n' 'forward 0.60 0.00 0.00' 'reverse -0.40 0.00 0.00' \
                'lateral_left 0.00 0.25 0.00' 'lateral_right 0.00 -0.25 0.00' \
                'yaw_left 0.00 0.00 0.60' 'yaw_right 0.00 0.00 -0.60' \
                'mixed 0.40 0.20 0.40'
            ;;
        random_rough)
            printf '%s\n' 'forward 0.50 0.00 0.00' 'reverse -0.20 0.00 0.00' \
                'lateral_left 0.00 0.15 0.00' 'lateral_right 0.00 -0.15 0.00' \
                'yaw_left 0.00 0.00 0.45' 'yaw_right 0.00 0.00 -0.45' \
                'mixed 0.35 0.12 0.35'
            ;;
        pyramid_slope|inverted_pyramid_slope)
            printf '%s\n' 'forward 0.40 0.00 0.00' 'reverse -0.15 0.00 0.00' \
                'lateral_left 0.00 0.10 0.00' 'lateral_right 0.00 -0.10 0.00' \
                'yaw_left 0.00 0.00 0.35' 'yaw_right 0.00 0.00 -0.35' \
                'mixed 0.30 0.08 0.30'
            ;;
        pyramid_stairs|inverted_pyramid_stairs|boxes)
            printf '%s\n' 'forward_low 0.20 0.00 0.00' 'forward_high 0.45 0.00 0.00' \
                'forward_correct_left 0.30 0.06 0.00' 'forward_correct_right 0.30 -0.06 0.00' \
                'forward_turn_left 0.30 0.00 0.25' 'forward_turn_right 0.30 0.00 -0.25' \
                'forward_mixed 0.35 0.05 0.20'
            ;;
        *) return 1 ;;
    esac
}

cd "$PROJECT_DIR"
if [[ "$DRY_RUN" != 1 && -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to reuse existing output root: $OUTPUT_ROOT" >&2
    exit 1
fi

for terrain in "${terrains[@]}"; do
    iteration="${ITERATION[$terrain]}"
    checkpoint="$PROJECT_DIR/${RUN_DIR[$terrain]}/model_${iteration}.pt"
    [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
    while read -r label vx vy yaw; do
        for evaluation_seed in "${evaluation_seeds[@]}"; do
            output_dir="$OUTPUT_ROOT/${terrain}_model${iteration}_${label}_trainseed42_evalseed${evaluation_seed}"
            cmd=(
                "$PYTHON_BIN" scripts/project/evaluate_go2.py
                --task "Unitree-Go2-Velocity-TeacherV1-${TASK_SUFFIX[$terrain]}"
                --terrain_protocol teacher_v1_stratified
                --num_envs "$NUM_ENVS"
                --episodes_per_env "$EPISODES_PER_ENV"
                --seed "$evaluation_seed"
                --fixed_command_x "$vx"
                --fixed_command_y "$vy"
                --fixed_command_yaw "$yaw"
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
                "${cmd[@]}" 2>&1 | tee "$OUTPUT_ROOT/console/${terrain}_model${iteration}_${label}_trainseed42_evalseed${evaluation_seed}.console.log"
                echo "$terrain model$iteration $label evalseed=$evaluation_seed exit code: 0"
            fi
        done
    done < <(commands_for "$terrain")
done

if [[ "$DRY_RUN" == 1 ]]; then
    echo "Terrain Teacher v1 fixed-command matrix dry-run completed; no evaluation launched."
    exit 0
fi

find "$OUTPUT_ROOT" -type f ! -name all_fixed_command_evidence.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$OUTPUT_ROOT/all_fixed_command_evidence.sha256"
sha256sum "$OUTPUT_ROOT/all_fixed_command_evidence.sha256"
echo "Terrain Teacher v1 fixed-command matrix finished: $OUTPUT_ROOT"
