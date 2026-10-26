#!/usr/bin/env bash
# S5 / S7: train B0-sim or 1A-sim on GPUs 2,3 (N₀-VTLA：仿真实验.md).
#   usage: train.sh b0|1a_rule <task> [extra train.sh args, e.g. --num-train-steps=200]
#   env:   DATA_ROOT ($LEROBOT_ROOT)  ASSET_ID (task asset)  EXP_SUFFIX ("")
#          PHASE_A_STEPS (2000)  PHASE_RAMP_STEPS (500)  TRAIN_GPUS (2,3)  BATCH (32)
# Both variants use the same recipe: batch 32 on 2 GPUs (16/GPU), 20k steps, phase A 2000 + ramp 500.
set -euo pipefail
source "$(dirname "$0")/tasks.sh"
VARIANT=$1; TASK=$2; shift 2
SHORT=$(task_short "$TASK")
DATA_ROOT=${DATA_ROOT:-$LEROBOT_ROOT}
ASSET=${ASSET_ID:-$(task_asset "$TASK")}
TRAIN_GPUS=${TRAIN_GPUS:-2,3}
case "$VARIANT" in
  b0)      REPO=$N0VTLA;    export CONFIG_NAME=sim_dual_arm_tactile ;;
  1a_rule) REPO=$N0VTLA_1A; export CONFIG_NAME=sim_dual_1a_rule
           export VTLA_1A_STATS=$N0VTLA_1A/assets/sim_dual_1a_rule/$ASSET/tactile_contact_stats.json
           # 1A keeps the B0 action space: reuse the B0 norm stats computed in the main tree (S4)
           mkdir -p "$N0VTLA_1A/assets/sim_dual_1a_rule/$ASSET"
           cp -n "$N0VTLA/assets/sim_dual_arm_tactile/$ASSET/norm_stats.json" "$N0VTLA_1A/assets/sim_dual_1a_rule/$ASSET/" ;;
  *) echo "unknown variant $VARIANT" >&2; exit 1 ;;
esac
source /mnt/ssd1/guningquan/miniforge3/etc/profile.d/conda.sh
set +u; conda activate vtla; set -u   # conda activate.d scripts read unset variables
cd "$REPO"
export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS NPROC_PER_NODE=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)
export VTLA_DATASET_PATH=$DATA_ROOT/$TASK
export VTLA_ASSET_ID=$ASSET
export VTLA_PRETRAINED_CHECKPOINT=$BASE_CKPT
export VTLA_DEFAULT_PROMPT="$(task_prompt "$TASK")"
export VTLA_PHASE_A_STEPS=${PHASE_A_STEPS:-2000} VTLA_PHASE_RAMP_STEPS=${PHASE_RAMP_STEPS:-500}
export EXP_NAME=${VARIANT}sim_$SHORT${EXP_SUFFIX:-} WANDB_MODE=offline
echo "[train] repo=$REPO config=$CONFIG_NAME exp=$EXP_NAME data=$VTLA_DATASET_PATH asset=$VTLA_ASSET_ID" \
     "gpus=$CUDA_VISIBLE_DEVICES phaseA=$VTLA_PHASE_A_STEPS/$VTLA_PHASE_RAMP_STEPS extra=$*"
CHECK_ONLY=1 bash train.sh
bash train.sh --overwrite --batch-size=${BATCH:-32} "$@"
