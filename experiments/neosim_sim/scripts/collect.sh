#!/usr/bin/env bash
# S3: collect scripted-expert demos for one task on GPUs 2/3 (N₀-VTLA：仿真实验.md).
#   usage: collect.sh <task> [gpus (2,3)] [workers (4)] [task_config (collect_vtla)]
# parallel_collect_data.py splits the workers over the listed GPUs; start_seed -1 resumes from suc_map.txt.
set -euo pipefail
source "$(dirname "$0")/tasks.sh"
TASK=$1; GPUS=${2:-2,3}; WORKERS=${3:-4}; CFG=${4:-collect_vtla}
source /mnt/ssd1/guningquan/miniforge3/etc/profile.d/conda.sh
set +u; conda activate UniVTAC; set -u   # conda activate.d scripts read unset variables
cd "$NEOSIM"
export OMNI_KIT_ACCEPT_EULA=YES VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
# driver 595 reports maxMemoryAllocationSize = UINT64_MAX, which crashes Isaac Sim 4.5 RTX: clamp it
# with the opt-in user layer in sim_runs/vk_layer (~/.local/share/vulkan/implicit_layer.d)
export VK_CLAMP_MAXALLOC=1
mkdir -p "$SIM_RUNS/logs/collect"
bash parallel_collect.sh "$TASK" "$CFG" "$GPUS" "$WORKERS" > "$SIM_RUNS/logs/collect/${TASK}.log" 2>&1
echo "[collect] $TASK: $(ls "$RAW_ROOT/$TASK/$CFG/hdf5" 2>/dev/null | wc -l) episodes in $RAW_ROOT/$TASK/$CFG/hdf5"
