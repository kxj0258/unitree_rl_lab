#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT=/home/kxj/rl_ws/unitree_rl_lab
PYTHON=/home/kxj/miniforge3/envs/unitree_rl/bin/python
OUTPUT=${OUTPUT:-${REPO_ROOT}/evaluation/portfolio_go2_current_results_multicheckpoint_demo_v1_2026-08-28.mp4}
DRY_RUN=${DRY_RUN:-0}

cd "${REPO_ROOT}" || exit $?

ARGS=(--output "${OUTPUT}")
if [[ "${DRY_RUN}" == "1" ]]; then
    ARGS+=(--dry_run)
fi

set +e
"${PYTHON}" scripts/project/compose_go2_current_results_demo_v1.py "${ARGS[@]}"
STATUS=$?
set -e

echo "Composition exit code: ${STATUS}"
echo "Output: ${OUTPUT}"
exit "${STATUS}"
