#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
CHECKPOINT="${PROJECT_DIR}/logs/rsl_rl/unitree_go2_velocity_v5_stairs_specialist/2026-08-27_16-17-34_v5_perceptive_gru_stairs_specialist_formal_continue_from2200_trainseed42_env1024_addi2800_total5000/model_4999.pt"
EXPECTED_SHA256="47b42309c03450c88f27ceaf66ed393abfcc5a488e7a508a369055ed4cfcee50"
OUTPUT_DIR="${PROJECT_DIR}/evaluation/v5_stairs_headinglocked_model4999_trainseed42_evalseed42_vx035_heading0_inityaw0_2026-08-27_smoke_v1"
CONSOLE_LOG="${PROJECT_DIR}/../eval_v5_stairs_headinglocked_model4999_trainseed42_evalseed42_vx035_heading0_inityaw0_2026-08-27_smoke_v1.log"
DRY_RUN="${DRY_RUN:-0}"

if [[ $# -ne 0 ]]; then
    echo "This wrapper takes no positional arguments." >&2
    exit 2
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1: $DRY_RUN" >&2
    exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi
actual_sha256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
    echo "Checkpoint SHA-256 mismatch: $actual_sha256" >&2
    exit 1
fi
if [[ -e "$OUTPUT_DIR" || -e "$CONSOLE_LOG" ]]; then
    echo "Refusing to overwrite heading-locked evaluation artifacts" >&2
    exit 1
fi

cd "$PROJECT_DIR"
command=(
    python scripts/project/evaluate_go2.py
    --task Unitree-Go2-Velocity-V5-Stairs-Specialist
    --terrain_protocol v5_stairs_stratified
    --num_envs 32
    --episodes_per_env 2
    --seed 42
    --fixed_command_x 0.35
    --fixed_command_y 0.0
    --fixed_command_yaw 0.0
    --fixed_heading 0.0
    --fixed_initial_yaw 0.0
    --checkpoint "$CHECKPOINT"
    --output_dir "$OUTPUT_DIR"
    --device cuda:0
)

echo "Task:                     Unitree-Go2-Velocity-V5-Stairs-Specialist"
echo "Task variant:             v5_stairs_specialist_headinglocked_smoke"
echo "Training seed:            42"
echo "Evaluation scenario seed: 42"
echo "Checkpoint:               $CHECKPOINT"
echo "Checkpoint SHA-256:       $actual_sha256"
echo "Fixed command:            vx=0.35 vy=0; world heading=0; initial yaw=0"
echo "Expected episodes:        64"
echo "Output directory:         $OUTPUT_DIR"
echo "Console log:              $CONSOLE_LOG"

if [[ "$DRY_RUN" == "1" ]]; then
    printf 'Dry-run command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

set +e
"${command[@]}" 2>&1 | tee "$CONSOLE_LOG"
status=${PIPESTATUS[0]}
set -e
echo "Evaluation exit code: $status"
exit "$status"
