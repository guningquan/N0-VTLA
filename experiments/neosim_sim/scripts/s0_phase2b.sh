#!/usr/bin/env bash
# S0 phase 2b: repair after setup.sh step 5, then setup.sh steps 6-8 by hand.
# setup.sh failed because (1) Isaac Lab's "all rl-frameworks" install pulled the newest skrl/sb3,
# which upgraded torch to 2.14.1+cu130, and (2) flatdict (an isaaclab dependency) cannot build with
# setuptools 84 (no pkg_resources), so isaaclab core was never installed.
# Pins come from neosim/requirements.lock (pip freeze of the authors' known-good node).
set -euo pipefail
ROOT=/mnt/ssd2/guningquan/Programs_server
NEOSIM=$ROOT/N0-Foundation/neosim
WORK=$ROOT/sim_deps
TACEX=$NEOSIM/third_party/TacEx/source
source /mnt/ssd1/guningquan/miniforge3/etc/profile.d/conda.sh
conda activate UniVTAC
export CMAKE_TOOLCHAIN_FILE=$HOME/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake
export OMNI_KIT_ACCEPT_EULA=YES VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export TORCH_CUDA_ARCH_LIST=8.6 CMAKE_CUDA_ARCHITECTURES=86 CUDA_VISIBLE_DEVICES=2
CONS=$ROOT/sim_runs/constraints_neosim.txt
grep -v -E '^-e |@ file://|^torch_scatter @|^robomimic @' "$NEOSIM/requirements.lock" > "$CONS"

echo "[2b.1] remove the torch 2.14 / CUDA 13 stack"
pip uninstall -y torch torchvision triton cuda-toolkit cuda-bindings cuda-pathfinder \
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime nvidia-cudnn-cu13 nvidia-cufft \
  nvidia-cufile nvidia-curand nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 \
  nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx || true

echo "[2b.2] torch 2.7.0+cu128 (locked)"
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple -c "$CONS"

echo "[2b.3] RL frameworks at the locked versions"
pip install skrl==2.0.0 stable_baselines3==2.8.0 rl_games==1.6.1 rsl-rl-lib==2.3.3 -c "$CONS"

echo "[2b.4] isaaclab core (flatdict needs pkg_resources: setuptools<81, no build isolation)"
pip install "setuptools<81"
pip install flatdict==4.0.1 --no-build-isolation
pip install -e "$WORK/IsaacLab/source/isaaclab" -c "$CONS"

echo "[2b.5] curobo (setup.sh step 6; system CUDA 12.8 matches torch cu128)"
CUDA_HOME=/usr/local/cuda-12.8 pip install -e "$WORK/curobo" --no-build-isolation -c "$CONS"

echo "[2b.6] TacEx incl. libuipc build (setup.sh step 7)"
pip install -e "$TACEX/tacex_assets" -e "$TACEX/tacex" -e "$TACEX/tacex_uipc" -e "$TACEX/tacex_tasks" -c "$CONS"

echo "[2b.7] remaining deps (setup.sh step 8) + client deps"
pip install numpy==1.26.4 warp-lang==1.12.1 transformers==5.5.4 trimesh==4.11.5 \
  opencv-python==4.11.0.86 scipy==1.15.3 -c "$CONS"
pip install "https://data.pyg.org/whl/torch-2.8.0+cu128/torch_scatter-2.1.2+pt28cu128-cp310-cp310-linux_x86_64.whl" \
  || echo "WARN: torch_scatter wheel failed"
pip install pyzmq msgpack

python - <<'PY'
import torch, isaaclab, tacex_uipc, curobo
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu ok", torch.cuda.is_available())
PY
pip check || true
echo "PHASE2B_OK"
