#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
NUM_ENVS="${NUM_ENVS:-1024}"
TRAINING_SEED="${TRAINING_SEED:-42}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
DEVICE="${DEVICE:-cuda:0}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-5}"
CONSOLE_ROOT="${CONSOLE_ROOT:-${PROJECT_DIR}/logs/teacher_v3_console}"
RUN_STAGE="${RUN_STAGE:-formal}"
RUN_TAG="${RUN_TAG:-}"

for name in NUM_ENVS MAX_ITERATIONS GPU_POLL_SECONDS; do
    value="${!name}"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "$name must be positive" >&2; exit 2; }
done
[[ "$TRAINING_SEED" =~ ^[0-9]+$ ]] || { echo "TRAINING_SEED must be non-negative" >&2; exit 2; }
[[ "$RUN_STAGE" == "construction_smoke" || "$RUN_STAGE" == "formal" ]] || {
    echo "RUN_STAGE must be construction_smoke or formal" >&2
    exit 2
}
[[ "$RUN_TAG" =~ ^[A-Za-z0-9_-]*$ ]] || { echo "RUN_TAG contains invalid characters" >&2; exit 2; }
run_tag_suffix=""
[[ -z "$RUN_TAG" ]] || run_tag_suffix="_${RUN_TAG}"
final_iteration=$((MAX_ITERATIONS - 1))

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
variants=(
    "worstleg_symmetry_heightscan_anticircle"
    "worstleg_symmetry_heightscan_ascent_anticircle"
    "worstleg_symmetry_edgeaware_mixedlevel_heightscan_anticircle"
)

mkdir -p "$CONSOLE_ROOT"
cd "$PROJECT_DIR"

for index in 0 1 2; do
    task="${tasks[$index]}"
    terrain="${terrains[$index]}"
    experiment="${experiments[$index]}"
    variant="${variants[$index]}"
    run_name="teacher_v3_${terrain}_${variant}_fromscratch_${RUN_STAGE}_trainseed${TRAINING_SEED}_env${NUM_ENVS}_i${MAX_ITERATIONS}${run_tag_suffix}"
    experiment_root="${PROJECT_DIR}/logs/rsl_rl/${experiment}"
    console_log="${CONSOLE_ROOT}/${run_name}.console.log"
    gpu_log="${CONSOLE_ROOT}/${run_name}.gpu.csv"

    if find "$experiment_root" -mindepth 1 -maxdepth 1 -type d -name "*_${run_name}" -print -quit 2>/dev/null | grep -q .; then
        echo "Refusing to repeat existing V3 run: $run_name" >&2
        exit 1
    fi
    [[ ! -e "$console_log" && ! -e "$gpu_log" ]] || {
        echo "Refusing to overwrite V3 logs: $run_name" >&2
        exit 1
    }

    echo "================================================================"
    echo "Task type:                Teacher V3 ${RUN_STAGE} training $((index + 1))/3"
    echo "Task:                     $task"
    echo "Terrain:                  $terrain"
    echo "Variant:                  $variant"
    echo "Run name:                 $run_name"
    echo "Training seed:            $TRAINING_SEED"
    echo "Evaluation scenario seed: none"
    echo "Initialization:           random/from-scratch; resume forbidden"
    echo "Number of environments:   $NUM_ENVS"
    echo "Maximum iterations:       $MAX_ITERATIONS"
    echo "Experiment root:          $experiment_root"
    echo "Console log:              $console_log"
    echo "GPU log:                  $gpu_log"
    echo "Start time:               $(date --iso-8601=seconds)"
    echo "================================================================"

    nvidia-smi --query-gpu=timestamp,index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
        --format=csv --loop="$GPU_POLL_SECONDS" >"$gpu_log" 2>&1 &
    gpu_pid=$!
    cleanup_gpu() {
        if kill -0 "$gpu_pid" 2>/dev/null; then
            kill "$gpu_pid" 2>/dev/null || true
            wait "$gpu_pid" 2>/dev/null || true
        fi
    }
    trap cleanup_gpu EXIT INT TERM

    set +e
    systemd-inhibit --what=sleep --why="Go2 $terrain Teacher V3 $RUN_STAGE training" \
        ./unitree_rl_lab.sh --train --task "$task" --num_envs "$NUM_ENVS" \
        --seed "$TRAINING_SEED" --max_iterations "$MAX_ITERATIONS" \
        --run_name "$run_name" --experiment_name "$experiment" \
        --logger tensorboard --device "$DEVICE" 2>&1 | tee "$console_log"
    status=${PIPESTATUS[0]}
    set -e
    cleanup_gpu
    trap - EXIT INT TERM

    echo "$terrain Teacher V3 $RUN_STAGE training exit code: $status"
    echo "End time: $(date --iso-8601=seconds)"
    if [[ "$status" -ne 0 ]]; then
        echo "Stopping serial training before the next terrain." >&2
        exit "$status"
    fi
    sync
done

echo "================================================================"
echo "All three Teacher V3 $RUN_STAGE trainings completed successfully."
for index in 0 1 2; do
    experiment_root="${PROJECT_DIR}/logs/rsl_rl/${experiments[$index]}"
    run_name="teacher_v3_${terrains[$index]}_${variants[$index]}_fromscratch_${RUN_STAGE}_trainseed${TRAINING_SEED}_env${NUM_ENVS}_i${MAX_ITERATIONS}${run_tag_suffix}"
    run_dir="$(find "$experiment_root" -mindepth 1 -maxdepth 1 -type d -name "*_${run_name}" -print | sort | tail -n 1)"
    echo "Terrain: ${terrains[$index]}"
    echo "Output directory: $run_dir"
    echo "Final checkpoint: ${run_dir}/model_${final_iteration}.pt"
    sha256sum "${run_dir}/model_${final_iteration}.pt"
done
echo "Serial end time: $(date --iso-8601=seconds)"
echo "================================================================"
