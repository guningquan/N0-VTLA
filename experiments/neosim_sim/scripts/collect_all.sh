#!/usr/bin/env bash
# S3: collect the three tasks one after another on GPUs 2,3 (N₀-VTLA：仿真实验.md).
#   usage: collect_all.sh [workers (4)]
set -uo pipefail
cd "$(dirname "$0")"
WORKERS=${1:-4}
for TASK in dual_bowl_place_stack dual_cup_handover_place dual_screw_sleeve; do
  START=$(date +%s)
  bash collect.sh "$TASK" 2,3 "$WORKERS"
  echo "[collect_all] $TASK done in $(( $(date +%s) - START ))s"
done
