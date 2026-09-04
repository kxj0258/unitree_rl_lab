#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
PYTHON_BIN="${PYTHON_BIN:-/home/kxj/miniforge3/envs/unitree_rl/bin/python}"
VIDEO_LENGTH="${VIDEO_LENGTH:-400}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
CAMERA_MODE="${CAMERA_MODE:-asset_follow}"
DEVICE="${DEVICE:-cuda:0}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/evaluation/video_teacher_v1_forward_steering_command_domain_demo_v1}"
COMPOSED_DIR="$OUTPUT_ROOT/composed"

[[ -d "$PROJECT_DIR" ]] || { echo "Project directory not found: $PROJECT_DIR" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 1; }
[[ "$VIDEO_LENGTH" =~ ^[1-9][0-9]*$ ]] || { echo "VIDEO_LENGTH must be positive" >&2; exit 2; }
[[ "$WARMUP_STEPS" =~ ^[0-9]+$ && "$WARMUP_STEPS" -lt "$VIDEO_LENGTH" ]] || {
    echo "WARMUP_STEPS must be non-negative and smaller than VIDEO_LENGTH" >&2
    exit 2
}
[[ "$CAMERA_MODE" == "asset_follow" || "$CAMERA_MODE" == "terrain_fixed" ]] || {
    echo "CAMERA_MODE must be asset_follow or terrain_fixed" >&2; exit 2;
}
[[ "$DRY_RUN" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
if [[ "$DRY_RUN" != 1 && -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite existing output root: $OUTPUT_ROOT" >&2
    exit 1
fi

declare -A TASK_SUFFIX=(
    [flat]="Flat"
    [random_rough]="RandomRough"
    [pyramid_slope]="PyramidSlope"
    [inverted_pyramid_slope]="InvertedPyramidSlope"
    [pyramid_stairs]="PyramidStairs"
    [inverted_pyramid_stairs]="InvertedPyramidStairs"
    [boxes]="Boxes"
)
declare -A CHECKPOINT=(
    [flat]="logs/rsl_rl/unitree_go2_velocity_teacher_v1_flat/2026-08-29_16-48-29_teacher_v1_flat_fromscratch_trainseed42_env1024_i5000/model_4999.pt"
    [random_rough]="logs/rsl_rl/unitree_go2_velocity_teacher_v1_random_rough/2026-08-29_19-35-33_teacher_v1_random_rough_fromscratch_trainseed42_env1024_i5000/model_4999.pt"
    [pyramid_slope]="logs/rsl_rl/unitree_go2_velocity_teacher_v1_pyramid_slope/2026-08-29_22-31-02_teacher_v1_pyramid_slope_fromscratch_trainseed42_env1024_i5000/model_4000.pt"
    [inverted_pyramid_slope]="logs/rsl_rl/unitree_go2_velocity_teacher_v1_inverted_pyramid_slope/2026-08-30_01-28-47_teacher_v1_inverted_pyramid_slope_fromscratch_trainseed42_env1024_i5000/model_4000.pt"
    [pyramid_stairs]="logs/rsl_rl/unitree_go2_velocity_teacher_v1_pyramid_stairs/2026-08-30_04-15-42_teacher_v1_pyramid_stairs_fromscratch_trainseed42_env1024_i5000/model_4700.pt"
    [inverted_pyramid_stairs]="logs/rsl_rl/unitree_go2_velocity_teacher_v1_inverted_pyramid_stairs/2026-08-30_07-13-57_teacher_v1_inverted_pyramid_stairs_fromscratch_trainseed42_env1024_i5000/model_4999.pt"
    [boxes]="logs/rsl_rl/unitree_go2_velocity_teacher_v1_boxes/2026-08-30_10-23-44_teacher_v1_boxes_fromscratch_trainseed42_env1024_i5000/model_4999.pt"
)
declare -A EXPECTED_SHA=(
    [flat]="5695c610874c1577390bf98ac909cb4b161f3b4cef0905cd12930e7fd4e1edc5"
    [random_rough]="8d139c4be23994ca474b9876c41fd03740956cda016328b01aaa52fb454178da"
    [pyramid_slope]="076a3691b8dd91bf4ab918a7e0e4c20987e7319a352ab783853fcae67367a1a4"
    [inverted_pyramid_slope]="00b8653af4ea8a0a88b4c2e14518c7851564abaef82041536734f6fcc4a7c491"
    [pyramid_stairs]="0fcdad89af3c7f27a2ab97f1094b9e67b4aa5be84377bef8db54025215359e23"
    [inverted_pyramid_stairs]="588ce05a44a96ffee6f7a236bf96affd80552ea4886ad4fdf3a426aa76ed7aa2"
    [boxes]="d73843ab31d92c585ec26381142ffafcc5efea31cb4de07ca533f44e6edb8b38"
)

terrains=(flat random_rough pyramid_slope inverted_pyramid_slope pyramid_stairs inverted_pyramid_stairs boxes)

# Records: scenario terrain_level evaluation_seed vx yaw.  vy is always zero,
# vx is always positive, and yaw is nonzero only while moving forward.
scenarios_for() {
    case "$1" in
        flat)
            printf '%s\n' 'straight_low 0 42 0.15 0.00' 'straight_mid 5 42 0.40 0.00' \
                'straight_high 9 42 0.70 0.00' 'forward_left_gentle 5 42 0.35 0.30' \
                'forward_right_gentle 5 42 0.35 -0.30' 'forward_left_strong 9 42 0.30 0.70' \
                'forward_right_strong 9 42 0.30 -0.70' ;;
        random_rough)
            printf '%s\n' 'straight_low 0 42 0.15 0.00' 'straight_mid 5 42 0.35 0.00' \
                'straight_high 9 7 0.60 0.00' 'forward_left_gentle 5 42 0.30 0.25' \
                'forward_right_gentle 5 42 0.30 -0.25' 'forward_left_strong 9 42 0.25 0.50' \
                'forward_right_strong 9 123 0.25 -0.50' ;;
        pyramid_slope|inverted_pyramid_slope)
            printf '%s\n' 'straight_low 0 42 0.15 0.00' 'straight_mid 5 42 0.30 0.00' \
                'straight_high 9 42 0.50 0.00' 'forward_left_gentle 5 42 0.25 0.20' \
                'forward_right_gentle 5 42 0.25 -0.20' 'forward_left_strong 9 42 0.20 0.40' \
                'forward_right_strong 9 123 0.20 -0.40' ;;
        pyramid_stairs)
            printf '%s\n' 'straight_low 0 42 0.18 0.00' 'straight_mid 5 42 0.30 0.00' \
                'straight_high 2 7 0.45 0.00' 'forward_left_gentle 5 42 0.30 0.12' \
                'forward_right_gentle 2 7 0.30 -0.12' 'forward_left_strong 9 42 0.25 0.25' \
                'forward_right_strong 9 123 0.25 -0.25' ;;
        inverted_pyramid_stairs)
            printf '%s\n' 'straight_low 0 42 0.18 0.00' 'straight_mid 5 42 0.30 0.00' \
                'straight_high 9 42 0.45 0.00' 'forward_left_gentle 5 42 0.30 0.12' \
                'forward_right_gentle 5 42 0.30 -0.12' 'forward_left_strong 9 42 0.25 0.25' \
                'forward_right_strong 9 123 0.25 -0.25' ;;
        boxes)
            printf '%s\n' 'straight_low 0 42 0.18 0.00' 'straight_mid 5 42 0.30 0.00' \
                'straight_high 9 42 0.45 0.00' 'forward_left_gentle 5 42 0.30 0.12' \
                'forward_right_gentle 5 42 0.30 -0.12' 'forward_left_strong 9 42 0.25 0.25' \
                'forward_right_strong 9 123 0.25 -0.25' ;;
        *) return 1 ;;
    esac
}

cd "$PROJECT_DIR"
for terrain in "${terrains[@]}"; do
    checkpoint="$PROJECT_DIR/${CHECKPOINT[$terrain]}"
    [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
    actual_sha="$(sha256sum "$checkpoint" | awk '{print $1}')"
    [[ "$actual_sha" == "${EXPECTED_SHA[$terrain]}" ]] || {
        echo "Checkpoint SHA mismatch for $terrain" >&2
        echo "Expected: ${EXPECTED_SHA[$terrain]}" >&2
        echo "Actual:   $actual_sha" >&2
        exit 1
    }
done

if [[ "$DRY_RUN" != 1 ]]; then mkdir -p "$OUTPUT_ROOT/console"; fi
for terrain in "${terrains[@]}"; do
    checkpoint="$PROJECT_DIR/${CHECKPOINT[$terrain]}"
    while read -r scenario level evaluation_seed vx yaw; do
        output_dir="$OUTPUT_ROOT/$terrain/$scenario"
        cmd=(
            "$PYTHON_BIN" scripts/project/record_go2_multiterrain_demo.py
            --task "Unitree-Go2-Velocity-TeacherV1-${TASK_SUFFIX[$terrain]}"
            --checkpoint "$checkpoint"
            --terrain_type "$terrain"
            --terrain_level "$level"
            --training_seed 42
            --seed "$evaluation_seed"
            --video_length "$VIDEO_LENGTH"
            --warmup_steps "$WARMUP_STEPS"
            --command_x "$vx"
            --command_y 0.00
            --command_yaw "$yaw"
            --camera_mode "$CAMERA_MODE"
            --output_dir "$output_dir"
            --device "$DEVICE"
        )
        printf 'RECORD COMMAND:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        if [[ "$DRY_RUN" != 1 ]]; then
            set +e
            "${cmd[@]}" 2>&1 | tee "$OUTPUT_ROOT/console/${terrain}_${scenario}_evalseed${evaluation_seed}.console.log"
            status=${PIPESTATUS[0]}
            set -e
            echo "$terrain $scenario evalseed=$evaluation_seed recording exit code: $status"
            [[ "$status" -eq 0 ]] || exit "$status"
        fi
    done < <(scenarios_for "$terrain")
done

compose_cmd=(
    "$PYTHON_BIN" scripts/project/compose_go2_terrain_teacher_v1_demos.py
    --source_root "$OUTPUT_ROOT"
    --output_dir "$COMPOSED_DIR"
)
printf 'COMPOSE COMMAND:'
printf ' %q' "${compose_cmd[@]}"
printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then
    echo "Teacher v1 demo dry-run completed; no recording or composition launched."
    exit 0
fi

"${compose_cmd[@]}" 2>&1 | tee "$OUTPUT_ROOT/console/composition.console.log"
compose_status=${PIPESTATUS[0]}
echo "Composition exit code: $compose_status"
[[ "$compose_status" -eq 0 ]] || exit "$compose_status"

find "$OUTPUT_ROOT" -type f ! -name all_teacher_v1_demo_evidence.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$OUTPUT_ROOT/all_teacher_v1_demo_evidence.sha256"
sha256sum "$OUTPUT_ROOT/all_teacher_v1_demo_evidence.sha256"
echo "Teacher v1 recording and composition finished: $OUTPUT_ROOT"
