#!/usr/bin/env bash
# S0 phase 2c: restore the CUDA 12 runtime wheels (uninstalling the CUDA 13 wheels in 2b deleted files
# with shared names, e.g. nvidia/cudnn/lib/libcudnn.so.9 and nvidia/nccl/lib/libnccl.so.2), then
# finish setup.sh steps 6-8 (2b stopped at the curobo build).
set -euo pipefail
ROOT=/mnt/ssd2/guningquan/Programs_server
NEOSIM=$ROOT/N0-Foundation/neosim
WORK=$ROOT/sim_deps
TACEX=$NEOSIM/third_party/TacEx/source
CONS=$ROOT/sim_runs/constraints_neosim.txt
source /mnt/ssd1/guningquan/miniforge3/etc/profile.d/conda.sh
conda activate UniVTAC
export CMAKE_TOOLCHAIN_FILE=$HOME/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake
export OMNI_KIT_ACCEPT_EULA=YES VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export TORCH_CUDA_ARCH_LIST=8.6 CMAKE_CUDA_ARCHITECTURES=86 CUDA_VISIBLE_DEVICES=2

echo "[2c.1] force-reinstall the CUDA 12 runtime wheels torch 2.7.0+cu128 uses"
pip install --force-reinstall --no-deps \
  nvidia-cublas-cu12==12.8.3.14 nvidia-cuda-cupti-cu12==12.8.57 nvidia-cuda-nvrtc-cu12==12.8.61 \
  nvidia-cuda-runtime-cu12==12.8.57 nvidia-cudnn-cu12==9.7.1.26 nvidia-cufft-cu12==11.3.3.41 \
  nvidia-cufile-cu12==1.13.0.11 nvidia-curand-cu12==10.3.9.55 nvidia-cusolver-cu12==11.7.2.55 \
  nvidia-cusparse-cu12==12.5.7.53 nvidia-cusparselt-cu12==0.6.3 nvidia-nccl-cu12==2.26.2 \
  nvidia-nvjitlink-cu12==12.8.61 nvidia-nvtx-cu12==12.8.55 triton==3.3.0
python -c "import torch; x = torch.randn(64, 64, device='cuda'); print('torch ok', torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), float((x @ x).sum()) == float((x @ x).sum()))"

echo "[2c.2] curobo (setup.sh step 6; system CUDA 12.8 matches torch cu128)"
CUDA_HOME=/usr/local/cuda-12.8 pip install -e "$WORK/curobo" --no-build-isolation -c "$CONS"

echo "[2c.3] TacEx incl. libuipc build (setup.sh step 7)"
pip install -e "$TACEX/tacex_assets" -e "$TACEX/tacex" -e "$TACEX/tacex_uipc" -e "$TACEX/tacex_tasks" -c "$CONS"

echo "[2c.4] remaining deps (setup.sh step 8) + client deps"
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
echo "PHASE2C_OK"
