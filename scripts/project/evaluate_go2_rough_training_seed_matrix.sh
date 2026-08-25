#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAINING_SEED="${TRAINING_SEED:-}"
ROUGH_RUN="${ROUGH_RUN:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
NUM_ENVS="${NUM_ENVS:-64}"
EPISODES_PER_ENV="${EPISODES_PER_ENV:-2}"
EVAL_SEEDS="${EVAL_SEEDS:-7 42 123}"
MODELS="${MODELS:-2400 2900 2999}"
DEVICE="${DEVICE:-cuda:0}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a eval_seeds <<<"$EVAL_SEEDS"
read -r -a models <<<"$MODELS"

usage() {
    cat <<'EOF'
Evaluate one formal Go2 rough training seed with rough_stratified v1.

Required environment variables:
  TRAINING_SEED  Training seed represented by ROUGH_RUN.
  ROUGH_RUN      Formal training directory containing model_*.pt.

Defaults:
  OUTPUT_ROOT=evaluation/rough_stratified_v1_trainseed${TRAINING_SEED}_YYYY-MM-DD
  MODELS="2400 2900 2999"
  EVAL_SEEDS="7 42 123"
  NUM_ENVS=64
  EPISODES_PER_ENV=2
  DEVICE=cuda:0
  DRY_RUN=0

The script refuses to overwrite OUTPUT_ROOT. Training seed and evaluation
scenario seed are both encoded in every output name.
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
if [[ ! "$TRAINING_SEED" =~ ^[0-9]+$ ]]; then
    echo "TRAINING_SEED must be a non-negative integer: $TRAINING_SEED" >&2
    exit 2
fi
if [[ -z "$ROUGH_RUN" || ! -d "$ROUGH_RUN" ]]; then
    echo "ROUGH_RUN must be an existing formal training directory: $ROUGH_RUN" >&2
    exit 2
fi
if [[ -z "$OUTPUT_ROOT" ]]; then
    OUTPUT_ROOT="${PROJECT_DIR}/evaluation/rough_stratified_v1_trainseed${TRAINING_SEED}_$(date +%Y-%m-%d)"
fi
if [[ ! "$NUM_ENVS" =~ ^[1-9][0-9]*$ ]] || (( NUM_ENVS < 20 || NUM_ENVS % 2 != 0 )); then
    echo "NUM_ENVS must be an even integer of at least 20: $NUM_ENVS" >&2
    exit 2
fi
if [[ ! "$EPISODES_PER_ENV" =~ ^[1-9][0-9]*$ ]]; then
    echo "EPISODES_PER_ENV must be a positive integer: $EPISODES_PER_ENV" >&2
    exit 2
fi
if [[ ${#eval_seeds[@]} -eq 0 || ${#models[@]} -eq 0 ]]; then
    echo "MODELS and EVAL_SEEDS must be non-empty" >&2
    exit 2
fi
for seed in "${eval_seeds[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid evaluation seed: $seed" >&2; exit 2; }
done
for model in "${models[@]}"; do
    [[ "$model" =~ ^[0-9]+$ ]] || { echo "Invalid model iteration: $model" >&2; exit 2; }
    checkpoint="${ROUGH_RUN}/model_${model}.pt"
    [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
done

echo "Project:          $PROJECT_DIR"
echo "Training seed:    $TRAINING_SEED"
echo "Training run:     $ROUGH_RUN"
echo "Output root:      $OUTPUT_ROOT"
echo "Models:           ${models[*]}"
echo "Evaluation seeds: ${eval_seeds[*]}"
echo "Environments:     $NUM_ENVS"
echo "Episodes/env:     $EPISODES_PER_ENV"
echo "Device:           $DEVICE"
echo "Dry run:          $DRY_RUN"

if [[ "$DRY_RUN" == "1" ]]; then
    for model in "${models[@]}"; do
        for eval_seed in "${eval_seeds[@]}"; do
            run_name="rough_trainseed${TRAINING_SEED}_model${model}_evalseed${eval_seed}_env${NUM_ENVS}_ep${EPISODES_PER_ENV}"
            echo "PLAN training_seed=$TRAINING_SEED model=$model eval_seed=$eval_seed run=$run_name checkpoint=${ROUGH_RUN}/model_${model}.pt"
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
completed_runs=0

for model in "${models[@]}"; do
    checkpoint="${ROUGH_RUN}/model_${model}.pt"
    for eval_seed in "${eval_seeds[@]}"; do
        run_name="rough_trainseed${TRAINING_SEED}_model${model}_evalseed${eval_seed}_env${NUM_ENVS}_ep${EPISODES_PER_ENV}"
        output_dir="${OUTPUT_ROOT}/${run_name}"
        console_log="${OUTPUT_ROOT}/${run_name}.console.log"
        start_seconds="$(date +%s)"

        echo "START training_seed=$TRAINING_SEED model=$model eval_seed=$eval_seed"
        set +e
        if command -v systemd-inhibit >/dev/null 2>&1; then
            PYTHONUNBUFFERED=1 systemd-inhibit --what=sleep --why="Go2 rough evaluation ${run_name}" \
                "$PYTHON_BIN" scripts/project/evaluate_go2.py \
                --task Unitree-Go2-Velocity-Rough \
                --terrain_protocol rough_stratified \
                --checkpoint "$checkpoint" \
                --num_envs "$NUM_ENVS" \
                --episodes_per_env "$EPISODES_PER_ENV" \
                --seed "$eval_seed" \
                --output_dir "$output_dir" \
                --device "$DEVICE" >"$console_log" 2>&1
        else
            PYTHONUNBUFFERED=1 "$PYTHON_BIN" scripts/project/evaluate_go2.py \
                --task Unitree-Go2-Velocity-Rough \
                --terrain_protocol rough_stratified \
                --checkpoint "$checkpoint" \
                --num_envs "$NUM_ENVS" \
                --episodes_per_env "$EPISODES_PER_ENV" \
                --seed "$eval_seed" \
                --output_dir "$output_dir" \
                --device "$DEVICE" >"$console_log" 2>&1
        fi
        evaluation_status=$?
        set -e

        elapsed_seconds=$(( $(date +%s) - start_seconds ))
        echo "EXIT training_seed=$TRAINING_SEED model=$model eval_seed=$eval_seed status=$evaluation_status elapsed_s=$elapsed_seconds"
        if [[ $evaluation_status -ne 0 ]]; then
            tail -100 "$console_log"
            exit "$evaluation_status"
        fi
        completed_runs=$((completed_runs + 1))
    done
done

expected_runs=$(( ${#models[@]} * ${#eval_seeds[@]} ))
summary_count="$(find "$OUTPUT_ROOT" -mindepth 2 -maxdepth 2 -type f -name summary.json | wc -l)"
[[ "$completed_runs" -eq "$expected_runs" && "$summary_count" -eq "$expected_runs" ]]

echo "MATRIX_COMPLETE"
echo "completed_runs=$completed_runs"
echo "summary_count=$summary_count"
echo "output_root=$OUTPUT_ROOT"
