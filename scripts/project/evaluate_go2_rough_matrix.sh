#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_ENVS="${NUM_ENVS:-64}"
EPISODES_PER_ENV="${EPISODES_PER_ENV:-2}"
EVAL_SEEDS="${EVAL_SEEDS:-7 42 123}"
DEVICE="${DEVICE:-cuda:0}"
DRY_RUN="${DRY_RUN:-0}"

default_baseline_run="${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity/"
default_baseline_run+="2026-08-24_23-25-37_baseline_seed42_env2048_i3000"
default_rough_run="${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity_rough/"
default_rough_run+="2026-08-25_04-40-21_roughmix50_curriculum_formal_seed42_env2048_i3000"
BASELINE_RUN="${BASELINE_RUN:-$default_baseline_run}"
ROUGH_RUN="${ROUGH_RUN:-$default_rough_run}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/evaluation/rough_stratified_v1_$(date +%Y-%m-%d)}"

labels=(
    flat_model2999
    rough_model2400
    rough_model2900
    rough_model2999
)
checkpoints=(
    "${BASELINE_RUN}/model_2999.pt"
    "${ROUGH_RUN}/model_2400.pt"
    "${ROUGH_RUN}/model_2900.pt"
    "${ROUGH_RUN}/model_2999.pt"
)
read -r -a seeds <<<"$EVAL_SEEDS"

usage() {
    cat <<'EOF'
Run the formal Go2 rough_stratified checkpoint matrix.

Defaults:
  NUM_ENVS=64
  EPISODES_PER_ENV=2
  EVAL_SEEDS="7 42 123"
  DEVICE=cuda:0
  DRY_RUN=0

The fixed matrix compares the flat model_2999 control with rough
model_2400, model_2900, and model_2999. Each evaluator output directory and
console log is new; OUTPUT_ROOT must not already exist.

Examples:
  DRY_RUN=1 scripts/project/evaluate_go2_rough_matrix.sh
  OUTPUT_ROOT=evaluation/rough_stratified_v1_rerun_2026-08-26 \
    scripts/project/evaluate_go2_rough_matrix.sh
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
if [[ $# -ne 0 ]]; then
    usage >&2
    exit 2
fi
if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "Project directory not found: $PROJECT_DIR" >&2
    exit 1
fi
if [[ ! "$NUM_ENVS" =~ ^[1-9][0-9]*$ ]] || (( NUM_ENVS < 20 || NUM_ENVS % 2 != 0 )); then
    echo "NUM_ENVS must be an even integer of at least 20: $NUM_ENVS" >&2
    exit 2
fi
if [[ ! "$EPISODES_PER_ENV" =~ ^[1-9][0-9]*$ ]]; then
    echo "EPISODES_PER_ENV must be a positive integer: $EPISODES_PER_ENV" >&2
    exit 2
fi
if [[ ${#seeds[@]} -eq 0 ]]; then
    echo "EVAL_SEEDS must contain at least one seed" >&2
    exit 2
fi
for seed in "${seeds[@]}"; do
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
        echo "Evaluation seeds must be non-negative integers: $seed" >&2
        exit 2
    fi
done
for checkpoint in "${checkpoints[@]}"; do
    if [[ ! -f "$checkpoint" ]]; then
        echo "Missing checkpoint: $checkpoint" >&2
        exit 1
    fi
done

echo "Project:          $PROJECT_DIR"
echo "Output root:      $OUTPUT_ROOT"
echo "Environments:     $NUM_ENVS"
echo "Episodes/env:     $EPISODES_PER_ENV"
echo "Seeds:            ${seeds[*]}"
echo "Device:           $DEVICE"
echo "Dry run:          $DRY_RUN"

if [[ "$DRY_RUN" == "1" ]]; then
    for index in "${!labels[@]}"; do
        for seed in "${seeds[@]}"; do
            echo "PLAN label=${labels[$index]} seed=$seed checkpoint=${checkpoints[$index]}"
        done
    done
    exit 0
fi
if [[ "$DRY_RUN" != "0" ]]; then
    echo "DRY_RUN must be 0 or 1: $DRY_RUN" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite existing output root: $OUTPUT_ROOT" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"
cd "$PROJECT_DIR"

for index in "${!labels[@]}"; do
    label="${labels[$index]}"
    checkpoint="${checkpoints[$index]}"

    for seed in "${seeds[@]}"; do
        run_name="${label}_seed${seed}_env${NUM_ENVS}_ep${EPISODES_PER_ENV}"
        output_dir="${OUTPUT_ROOT}/${run_name}"
        console_log="${OUTPUT_ROOT}/${run_name}.console.log"
        start_seconds="$(date +%s)"

        echo
        echo "START label=$label seed=$seed"

        set +e
        PYTHONUNBUFFERED=1 "$PYTHON_BIN" scripts/project/evaluate_go2.py \
            --task Unitree-Go2-Velocity-Rough \
            --terrain_protocol rough_stratified \
            --checkpoint "$checkpoint" \
            --num_envs "$NUM_ENVS" \
            --episodes_per_env "$EPISODES_PER_ENV" \
            --seed "$seed" \
            --output_dir "$output_dir" \
            --device "$DEVICE" \
            >"$console_log" 2>&1
        evaluation_status=$?
        set -e

        elapsed_seconds=$(( $(date +%s) - start_seconds ))
        echo "EXIT label=$label seed=$seed status=$evaluation_status elapsed_s=$elapsed_seconds"

        if [[ $evaluation_status -ne 0 ]]; then
            tail -100 "$console_log"
            exit "$evaluation_status"
        fi

        "$PYTHON_BIN" - "$output_dir/summary.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    summary = json.load(stream)
by_type = summary["terrain_metrics"]["by_type"]
flat = by_type["flat"]
rough = by_type["random_rough"]
print(
    "SUMMARY"
    f" episodes={summary['completed_episodes']}"
    f" overall_xy={summary['mean_xy_velocity_error_mps']:.6f}"
    f" flat_xy={flat['mean_xy_velocity_error_mps']:.6f}"
    f" rough_xy={rough['mean_xy_velocity_error_mps']:.6f}"
    f" overall_timeout={summary['time_out_rate']:.6f}"
    f" rough_timeout={rough['time_out_rate']:.6f}"
    f" bad_orientation={summary['bad_orientation_rate']:.6f}"
    f" throughput={summary['transitions_per_second']:.1f}"
)
PY
    done
done

expected_runs=$(( ${#labels[@]} * ${#seeds[@]} ))
summary_count="$(find "$OUTPUT_ROOT" -mindepth 2 -maxdepth 2 -name summary.json -type f | wc -l)"
if [[ "$summary_count" -ne "$expected_runs" ]]; then
    echo "Expected $expected_runs summaries, found $summary_count" >&2
    exit 1
fi

echo
echo "MATRIX_COMPLETE"
echo "summary_count=$summary_count"
echo "output_root=$OUTPUT_ROOT"
