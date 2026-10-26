#!/usr/bin/env bash
# S0 phase 2d: libuipc toolchain + tacex_uipc, then the rest of setup.sh step 8.
# setup.sh step 2 runs "conda create -y -n UniVTAC", which on an existing env answers "remove it?"
# with yes: the env was recreated and lost the CMake 3.26 / GCC 11.4 / CUDA 12.4 toolchain that
# phase 1 installed from libuipc's env.yaml. The user profile also puts ~/.local/bin and
# /usr/local/cuda-12.8/bin ahead of the env, so the build PATH is set explicitly.
set -euo pipefail
ROOT=/mnt/ssd2/guningquan/Programs_server
NEOSIM=$ROOT/N0-Foundation/neosim
TACEX=$NEOSIM/third_party/TacEx/source
CONS=$ROOT/sim_runs/constraints_neosim.txt
source /mnt/ssd1/guningquan/miniforge3/etc/profile.d/conda.sh

echo "[2d.1] restore the libuipc toolchain from env.yaml"
conda env update -n UniVTAC --file "$TACEX/tacex_uipc/libuipc/conda/env.yaml"
set +u; conda activate UniVTAC; set -u   # the conda-forge binutils/gcc activate.d scripts read unset variables
export PATH=$CONDA_PREFIX/bin:$PATH
unset CUDA_HOME
export CMAKE_TOOLCHAIN_FILE=$HOME/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake
export CMAKE_CUDA_ARCHITECTURES=86 CUDA_VISIBLE_DEVICES=2
export VCPKG_OVERLAY_PORTS=$ROOT/sim_runs/vcpkg-overlays   # tinygltf 2.9.3 with the current archive hash
export OMNI_KIT_ACCEPT_EULA=YES VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
echo "cmake: $(which cmake) $(cmake --version | head -1)"
echo "nvcc:  $(which nvcc) $(nvcc --version | tail -1)"
echo "gcc:   $(which gcc) $(gcc --version | head -1)"
python -c "import torch; print('torch still ok', torch.__version__, torch.cuda.is_available())"

echo "[2d.2] pre-build libuipc with 32 jobs (tacex_uipc/setup.py hard-codes -j4; same args otherwise)"
mkdir -p "$TACEX/tacex_uipc/build"
cd "$TACEX/tacex_uipc/build"
cmake ../libuipc -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DUIPC_BUILD_PYBIND=1 -DUIPC_DEV_MODE=1 \
  -DUIPC_BUILD_GUI=0 -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release -DVCPKG_OVERLAY_PORTS=$VCPKG_OVERLAY_PORTS
cmake --build . -j32

echo "[2d.3] tacex_uipc editable install (incremental after the pre-build)"
cd "$NEOSIM"
pip install -e "$TACEX/tacex_uipc" -c "$CONS"

echo "[2d.4] remaining deps (setup.sh step 8) + client deps"
pip install numpy==1.26.4 warp-lang==1.12.1 transformers==5.5.4 trimesh==4.11.5 \
  opencv-python==4.11.0.86 scipy==1.15.3 -c "$CONS"
pip install "https://data.pyg.org/whl/torch-2.8.0+cu128/torch_scatter-2.1.2+pt28cu128-cp310-cp310-linux_x86_64.whl" \
  || echo "WARN: torch_scatter wheel failed"
pip install pyzmq msgpack

python - <<'PY'
import torch, isaaclab, tacex_uipc, curobo
import uipc
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu ok", torch.cuda.is_available())
print("uipc", getattr(uipc, "__file__", "?"))
PY
pip check || true
echo "PHASE2D_OK"
