#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
PYTHON_EXE="${PYTHON_EXE:-/home/kxj/miniforge3/envs/unitree_rl/bin/python}"
CHECKPOINT="${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity_v5_stairs_ascending_remediation/2026-08-27_23-01-17_v5_stairs_ascending_remediation_formal_v1_from4999_trainseed42_env1024_i1200/model_900.pt"
EXPECTED_SHA256="57c66a27fb3be566ad28edec6e567bfc1f8eddea8db3115e0ed8009c9f38d1a3"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/evaluation/video_v5_stairs_headinglocked_ascendingremediation_model900_trainseed42_evalseed42_inverted_level7_vx035_heading0_inityaw0_20s_2026-08-28_v1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ $# -ne 0 ]]; then
    echo "This wrapper takes no positional arguments." >&2
    exit 2
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1: $DRY_RUN" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_EXE" || ! -f "$CHECKPOINT" ]]; then
    echo "Python executable or checkpoint not found." >&2
    exit 1
fi
actual_sha256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
    echo "Checkpoint SHA-256 mismatch: $actual_sha256" >&2
    exit 1
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite existing output: $OUTPUT_ROOT" >&2
    exit 1
fi

command=(
    "$PYTHON_EXE" scripts/project/record_go2_multiterrain_demo.py
    --task Unitree-Go2-Velocity-V5-Stairs-Specialist
    --checkpoint "$CHECKPOINT"
    --terrain_type inverted_pyramid_stairs
    --terrain_level 7
    --training_seed 42
    --seed 42
    --video_length 1000
    --warmup_steps 100
    --command_x 0.35
    --command_y 0.0
    --command_yaw 0.0
    --fixed_heading 0.0
    --fixed_initial_yaw 0.0
    --camera_mode asset_follow
    --output_dir "$OUTPUT_ROOT/inverted_pyramid_stairs_level7_vx035_heading0_inityaw0"
    --device cuda:0
)

echo "Task:                     Unitree-Go2-Velocity-V5-Stairs-Specialist"
echo "Task variant:             v5_stairs_headinglocked_ascending_video_diagnostic"
echo "Training seed:            42"
echo "Evaluation scenario seed: 42"
echo "Checkpoint:               $CHECKPOINT"
echo "Checkpoint SHA-256:       $actual_sha256"
echo "Terrain:                  inverted_pyramid_stairs level 7"
echo "Command:                  vx=0.35 vy=0 heading=0 initial_yaw=0"
echo "Output root:              $OUTPUT_ROOT"

if [[ "$DRY_RUN" == "1" ]]; then
    printf 'Dry-run command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "$OUTPUT_ROOT"
cd "$PROJECT_DIR"
console_log="$OUTPUT_ROOT/recording.console.log"
set +e
"${command[@]}" 2>&1 | tee "$console_log"
status=${PIPESTATUS[0]}
set -e
echo "Recording exit code (inverted_pyramid_stairs): $status"
if [[ "$status" -ne 0 ]]; then
    exit "$status"
fi

find "$OUTPUT_ROOT" -type f -printf '%s %P\n' | sort -k2 > "$OUTPUT_ROOT/inventory.txt"
find "$OUTPUT_ROOT" -type f ! -name 'final_artifacts.sha256' -print0 \
    | sort -z | xargs -0 sha256sum > "$OUTPUT_ROOT/final_artifacts.sha256"
echo "[V5_STAIRS_ASCENDING_MODEL900_LEVEL7] completed successfully: $OUTPUT_ROOT"
