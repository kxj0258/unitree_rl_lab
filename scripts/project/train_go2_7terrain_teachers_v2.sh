#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${UNITREE_RL_LAB_PATH:-/home/kxj/rl_ws/unitree_rl_lab}"
DRY_RUN="${DRY_RUN:-0}"
terrains=(flat random_rough pyramid_slope inverted_pyramid_slope pyramid_stairs inverted_pyramid_stairs boxes)

cd "$PROJECT_DIR"
for terrain in "${terrains[@]}"; do
    TERRAIN="$terrain" DRY_RUN=1 scripts/project/train_go2_terrain_teacher_v2.sh
done
if [[ "$DRY_RUN" == 1 ]]; then
    echo "Seven Teacher v2 preflight completed; no training launched."
    exit 0
fi
for terrain in "${terrains[@]}"; do
    TERRAIN="$terrain" DRY_RUN=0 scripts/project/train_go2_terrain_teacher_v2.sh
done
