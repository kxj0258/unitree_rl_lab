#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
PYTHON_EXE="${PYTHON_EXE:-/home/kxj/miniforge3/envs/unitree_rl/bin/python}"
EXECUTE="${EXECUTE:-0}"
DEVICE="${DEVICE:-cuda:0}"

BASE_RUN="2026-08-25_22-53-12_multiterrain_v3_challenging_noheightscan_prototype_resume_from2396_seed42_env1024_addi600_totalapprox3000"
BASE_CHECKPOINT="model_2995.pt"
BASELINE_RUN="2026-08-26_04-32-39_multiterrain_v3_challenging_noheightscan_baseline_continue_from2995_seed42_env1024_addi500"
TRACKING_RUN="2026-08-26_04-58-48_multiterrain_v3_trackingaware_thr065_continue_from2995_seed42_env1024_addi500"
EXPERIMENT_ROOT="${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity_multiterrain"
SELECTED_CHECKPOINT="${EXPERIMENT_ROOT}/${BASELINE_RUN}/model_3300.pt"

usage() {
    cat <<'EOF'
Reproduce or verify the archived Go2 MultiTerrain Stage-2 workflow.

Safety: commands are printed by default. Set EXECUTE=1 explicitly to run them.

Usage:
  scripts/project/reproduce_go2_multiterrain_stage2.sh <command>

Commands:
  baseline-train       Continue the original model_2995 with survival curriculum for 500 iterations.
  tracking-train       Continue the same model_2995 with tracking-aware curriculum for 500 iterations.
  baseline-eval        Evaluate CHECKPOINT (default: selected baseline model_3300) at seeds 7/42/123.
  tracking-eval        Evaluate archived tracking-aware model_3300/3400/3494 at scenario seed 42.
  multiterrain-video   Record seven 20-second level-5 clips with asset-follow camera.
  rough-video          Record three rough training-seed clips with the same video protocol.
  verify-archive       Verify the selected Stage-2 SHA-256 manifest.

Required unique naming:
  REPRO_TAG=<label> is required for training commands.
  OUTPUT_ROOT=<new path> is required for evaluation and video commands.

Examples:
  REPRO_TAG=audit1 scripts/project/reproduce_go2_multiterrain_stage2.sh baseline-train
  EXECUTE=1 REPRO_TAG=audit1 scripts/project/reproduce_go2_multiterrain_stage2.sh baseline-train
  OUTPUT_ROOT=evaluation/repro_mt_eval scripts/project/reproduce_go2_multiterrain_stage2.sh baseline-eval
EOF
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

run_command() {
    print_command "$@"
    if [[ "$EXECUTE" == "1" ]]; then
        "$@"
    fi
}

require_new_output_root() {
    if [[ -z "${OUTPUT_ROOT:-}" ]]; then
        echo "OUTPUT_ROOT is required" >&2
        exit 2
    fi
    if [[ -e "$OUTPUT_ROOT" ]]; then
        echo "Refusing existing OUTPUT_ROOT: $OUTPUT_ROOT" >&2
        exit 1
    fi
}

require_repro_tag() {
    if [[ -z "${REPRO_TAG:-}" || ! "$REPRO_TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "REPRO_TAG is required and must use only letters, digits, dot, underscore, or hyphen" >&2
        exit 2
    fi
}

if [[ "$EXECUTE" != "0" && "$EXECUTE" != "1" ]]; then
    echo "EXECUTE must be 0 or 1" >&2
    exit 2
fi
if [[ ! -d "$PROJECT_DIR" || ! -x "$PYTHON_EXE" ]]; then
    echo "Project directory or Python executable is unavailable" >&2
    exit 1
fi

cd "$PROJECT_DIR"
command="${1:-}"
case "$command" in
    baseline-train)
        require_repro_tag
        run_command env \
            TASK_VARIANT=survival NUM_ENVS=1024 SEED=42 MAX_ITERATIONS=500 \
            RUN_STAGE=baseline_continue500 \
            MULTITERRAIN_RUN_NAME="multiterrain_v3_survival_continue_from2995_seed42_env1024_addi500_repro_${REPRO_TAG}" \
            RESUME_RUN="$BASE_RUN" RESUME_CHECKPOINT="$BASE_CHECKPOINT" \
            scripts/project/train_go2_multiterrain.sh
        ;;
    tracking-train)
        require_repro_tag
        run_command env \
            TASK_VARIANT=tracking_aware NUM_ENVS=1024 SEED=42 MAX_ITERATIONS=500 \
            RUN_STAGE=tracking_aware_continue500 \
            MULTITERRAIN_RUN_NAME="multiterrain_v3_trackingaware_thr065_continue_from2995_seed42_env1024_addi500_repro_${REPRO_TAG}" \
            RESUME_RUN="$BASE_RUN" RESUME_CHECKPOINT="$BASE_CHECKPOINT" \
            scripts/project/train_go2_multiterrain.sh
        ;;
    baseline-eval)
        require_new_output_root
        checkpoint="${CHECKPOINT:-$SELECTED_CHECKPOINT}"
        for scenario_seed in 7 42 123; do
            output_dir="${OUTPUT_ROOT}/baseline_model3300_trainseed42_evalseed${scenario_seed}"
            run_command "$PYTHON_EXE" scripts/project/evaluate_go2.py \
                --task Unitree-Go2-Velocity-MultiTerrain \
                --terrain_protocol multiterrain_stratified \
                --num_envs 70 --episodes_per_env 2 --seed "$scenario_seed" \
                --checkpoint "$checkpoint" --output_dir "$output_dir" --device "$DEVICE"
        done
        ;;
    tracking-eval)
        require_new_output_root
        for checkpoint_id in 3300 3400 3494; do
            checkpoint="${EXPERIMENT_ROOT}/${TRACKING_RUN}/model_${checkpoint_id}.pt"
            output_dir="${OUTPUT_ROOT}/trackingaware_model${checkpoint_id}_trainseed42_evalseed42"
            run_command "$PYTHON_EXE" scripts/project/evaluate_go2.py \
                --task Unitree-Go2-Velocity-MultiTerrain \
                --terrain_protocol multiterrain_stratified \
                --num_envs 70 --episodes_per_env 2 --seed 42 \
                --checkpoint "$checkpoint" --output_dir "$output_dir" --device "$DEVICE"
        done
        ;;
    multiterrain-video)
        require_new_output_root
        run_command env \
            CHECKPOINT="$SELECTED_CHECKPOINT" OUTPUT_ROOT="$OUTPUT_ROOT" \
            TRAINING_SEED=42 EVALUATION_SEED=42 TERRAIN_LEVEL=5 \
            VIDEO_LENGTH=1000 WARMUP_STEPS=100 COMMAND_X=0.35 CAMERA_MODE=asset_follow \
            DEVICE="$DEVICE" scripts/project/record_go2_multiterrain_demo.sh
        ;;
    rough-video)
        require_new_output_root
        run_command env \
            OUTPUT_ROOT="$OUTPUT_ROOT" EVALUATION_SEED=42 TERRAIN_LEVEL=5 \
            VIDEO_LENGTH=1000 WARMUP_STEPS=100 COMMAND_X=0.35 CAMERA_MODE=asset_follow \
            DEVICE="$DEVICE" scripts/project/record_go2_rough_multitraining_demo.sh
        ;;
    verify-archive)
        run_command sha256sum -c doc/repro/go2_multiterrain_stage2_final_artifacts_2026-08-26.sha256
        ;;
    -h|--help|help|"")
        usage
        ;;
    *)
        echo "Unknown command: $command" >&2
        usage >&2
        exit 2
        ;;
esac

if [[ "$EXECUTE" == "0" && -n "$command" && "$command" != "help" && "$command" != "-h" && "$command" != "--help" ]]; then
    echo "[DRY RUN] Set EXECUTE=1 to execute the printed command(s)."
fi
