#!/usr/bin/env bash
# S0 phase 1 (N₀-VTLA：仿真实验.md, S0 steps 1-4a): TacEx source, assets, vcpkg, conda env + libuipc toolchain.
set -uo pipefail
ROOT=/mnt/ssd2/guningquan/Programs_server
NEOSIM=$ROOT/N0-Foundation/neosim
LOGS=$ROOT/sim_runs/logs
source /mnt/ssd1/guningquan/miniforge3/etc/profile.d/conda.sh

vcpkg_step() {
  mkdir -p "$HOME/Toolchain"
  [ -d "$HOME/Toolchain/vcpkg/.git" ] || git clone https://github.com/microsoft/vcpkg.git "$HOME/Toolchain/vcpkg"
  [ -x "$HOME/Toolchain/vcpkg/vcpkg" ] || "$HOME/Toolchain/vcpkg/bootstrap-vcpkg.sh" -disableMetrics
}
env_step() {
  conda env list | awk '{print $1}' | grep -qx UniVTAC || conda create -n UniVTAC python=3.10 -y
}

( vcpkg_step ) > "$LOGS/s0_vcpkg.log" 2>&1 & P_VCPKG=$!
( env_step )   > "$LOGS/s0_conda_create.log" 2>&1 & P_ENV=$!

# TacEx source (libuipc is a plain directory in this branch, so a shallow clone is enough)
{
  [ -d "$ROOT/UniVTAC-NeoSim/.git" ] || git clone --depth 1 -b NeoSim https://github.com/anlorla/UniVTAC.git "$ROOT/UniVTAC-NeoSim"
  mkdir -p "$NEOSIM/third_party"
  [ -d "$NEOSIM/third_party/TacEx" ] || cp -r "$ROOT/UniVTAC-NeoSim/third_party/TacEx" "$NEOSIM/third_party/"
} > "$LOGS/s0_tacex.log" 2>&1 || { echo "TacEx step failed"; exit 1; }

# Official assets (HF NeoteAI/NeoSim-Assets)
{
  conda activate vtla
  HF_HUB_OFFLINE=0 hf download NeoteAI/NeoSim-Assets --repo-type dataset --local-dir "$ROOT/NeoSim-Assets"
  conda deactivate
  [ -d "$NEOSIM/assets" ] || cp -r "$ROOT/NeoSim-Assets/assets" "$NEOSIM/"
  cp -r "$ROOT/NeoSim-Assets/tacex_data/." "$NEOSIM/third_party/TacEx/source/tacex_assets/tacex_assets/data/"
} > "$LOGS/s0_assets.log" 2>&1 || { echo "assets step failed"; exit 1; }

wait $P_ENV   || { echo "conda create failed"; exit 1; }
# CMake 3.26 / GCC 11.4 / CUDA 12.4 for the libuipc build (needs TacEx source in place)
conda env update -n UniVTAC --file "$NEOSIM/third_party/TacEx/source/tacex_uipc/libuipc/conda/env.yaml" \
  > "$LOGS/s0_conda_update.log" 2>&1 || { echo "conda env update failed"; exit 1; }
wait $P_VCPKG || { echo "vcpkg failed"; exit 1; }
echo "PHASE1_OK"
