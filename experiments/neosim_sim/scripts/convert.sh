#!/usr/bin/env bash
# S4: convert one task's NeoSim HDF5 episodes to LeRobot and compute its norm stats (N₀-VTLA：仿真实验.md).
#   usage: convert.sh <task> [src_config (collect_vtla)]
set -euo pipefail
source "$(dirname "$0")/tasks.sh"
TASK=$1; SRC_CFG=${2:-collect_vtla}
ASSET=$(task_asset "$TASK")
source /mnt/ssd1/guningquan/miniforge3/etc/profile.d/conda.sh
set +u; conda activate vtla; set -u   # conda activate.d scripts read unset variables
cd "$N0VTLA"
export PYTHONPATH=$N0VTLA
python scripts/convert_neosim_to_lerobot.py \
  --src "$RAW_ROOT/$TASK/$SRC_CFG/hdf5" \
  --dst "$LEROBOT_ROOT/$TASK" \
  --prompt "$(task_prompt "$TASK")" --fps 60 --overwrite
# norm stats -> assets/sim_dual_arm_tactile/<asset>/norm_stats.json (train.sh's offline env)
export VTLA_DATASET_PATH=$LEROBOT_ROOT/$TASK VTLA_ASSET_ID=$ASSET
export N0VTLA_DATA_HOME=$N0VTLA/models HF_HOME=$N0VTLA/models/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/compute_norm_stats.py --config-name sim_dual_arm_tactile
ls -l "assets/sim_dual_arm_tactile/$ASSET/norm_stats.json"
