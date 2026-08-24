#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    echo "Usage: $0 CHECKPOINT OUTPUT_DIR [follow|world|grid] [NUM_ENVS] [SEED] [VIDEO_LENGTH]"
    echo
    echo "Examples:"
    echo "  $0 /path/model_2400.pt evaluation/model2400_follow follow 1 42 1000"
    echo "  $0 /path/model_2400.pt evaluation/model2400_grid9 grid 9 42 1000"
}

if [[ $# -lt 2 ]]; then
    usage
    exit 2
fi

checkpoint="$(realpath "$1")"
output_dir="$(realpath -m "$2")"
camera_mode="${3:-follow}"
num_envs="${4:-1}"
seed="${5:-42}"
video_length="${6:-1000}"

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"

if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint not found: $checkpoint" >&2
    exit 1
fi

case "$camera_mode" in
    follow|world|grid) ;;
    *)
        echo "Invalid camera mode: $camera_mode" >&2
        usage
        exit 2
        ;;
esac

if [[ -e "$output_dir" ]]; then
    echo "Refusing to reuse existing output directory: $output_dir" >&2
    exit 1
fi

mkdir -p "$output_dir"
checkpoint_copy="$output_dir/$(basename "$checkpoint")"
cp "$checkpoint" "$checkpoint_copy"

sha256sum "$checkpoint" "$checkpoint_copy"

cd "$PROJECT_DIR"

set +e
./unitree_rl_lab.sh --play \
    --task Unitree-Go2-Velocity \
    --num_envs "$num_envs" \
    --seed "$seed" \
    --checkpoint "$checkpoint_copy" \
    --video \
    --video_camera "$camera_mode" \
    --video_length "$video_length" \
    --headless \
    2>&1 | tee "$output_dir/video_play.log"

play_status=${PIPESTATUS[0]}
set -e

video_file="$(find "$output_dir" -type f -name '*.mp4' -print -quit)"

if grep -qE 'Traceback|ModuleNotFoundError|Segmentation fault' "$output_dir/video_play.log"; then
    echo "Video run contains a fatal error marker." >&2
    exit 1
fi

if [[ -z "$video_file" || ! -s "$video_file" ]]; then
    echo "No non-empty MP4 was generated." >&2
    exit 1
fi

echo "Wrapper exit code: $play_status"
echo "Video: $video_file"
sha256sum "$video_file"

find "$output_dir" -maxdepth 5 -type f \
    -printf '%TY-%Tm-%Td %TH:%TM:%TS %12s %P\n' | sort
