#!/usr/bin/env bash
# S0 phase 2 (N₀-VTLA：仿真实验.md, S0 steps 4b-5): NeoSim setup.sh, client deps, self-check.
set -uo pipefail
ROOT=/mnt/ssd2/guningquan/Programs_server
NEOSIM=$ROOT/N0-Foundation/neosim
source /mnt/ssd1/guningquan/miniforge3/etc/profile.d/conda.sh

export CMAKE_TOOLCHAIN_FILE=$HOME/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake
export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export TORCH_CUDA_ARCH_LIST=8.6          # RTX A6000 only: keeps the cuRobo build short
export CUDA_VISIBLE_DEVICES=2            # only GPUs 2/3 are ours
export CUDA_HOME=/usr/local/cuda-12.8    # cuRobo: standard layout, matches torch cu128 (the conda CUDA keeps headers under targets/)
export CMAKE_CUDA_ARCHITECTURES=86       # libuipc: RTX A6000

cd "$NEOSIM"
SKIP_APT=1 WORK=$ROOT/sim_deps bash setup.sh || { echo "SETUP_SH_FAILED"; exit 1; }

conda activate UniVTAC
pip install pyzmq msgpack || { echo "PIP_CLIENT_DEPS_FAILED"; exit 1; }
python - <<'PY' || { echo "SELFCHECK_FAILED"; exit 1; }
import torch, isaaclab, tacex_uipc
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY
echo "PHASE2_OK"
