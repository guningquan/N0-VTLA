#!/usr/bin/env bash
# S2 / S5 / S8: evaluate one policy in NeoSim (N₀-VTLA：仿真实验.md).
#   usage: eval.sh released|b0|1a_rule <task>
#   env:   SERVER_GPU (3) CLIENT_GPU (2) PORT (5557) TOTAL (100) START_SEED (1000000)
#          TASK_CONFIG (fasteval) STEPS_PER_ACTION (2) CKPT_STEP (20000) TAG_SUFFIX ("")
#          EXP_SUFFIX ("", as in train.sh) ASSET_ID (task asset)
# One server per client: the server keeps a single per-episode tactile baseline.
set -euo pipefail
source "$(dirname "$0")/tasks.sh"
VARIANT=$1; TASK=$2
SHORT=$(task_short "$TASK"); PROMPT=$(task_prompt "$TASK")
SERVER_GPU=${SERVER_GPU:-3}; CLIENT_GPU=${CLIENT_GPU:-2}; PORT=${PORT:-5557}
TOTAL=${TOTAL:-100}; START_SEED=${START_SEED:-1000000}; TASK_CONFIG=${TASK_CONFIG:-fasteval}
STEPS_PER_ACTION=${STEPS_PER_ACTION:-2}; CKPT_STEP=${CKPT_STEP:-20000}
unset VTLA_1A_STATS
case "$VARIANT" in
  released) REPO=$N0VTLA; CONFIG=sim_dual_arm_tactile
            CKPT=$N0VTLA/checkpoints/n0_VTLA_dual_bowl_place_stack; ASSET=n0_dual_bowl_place_stack_norm ;;
  b0)       REPO=$N0VTLA; CONFIG=sim_dual_arm_tactile
            CKPT=$N0VTLA/checkpoints/sim_dual_arm_tactile/b0sim_$SHORT${EXP_SUFFIX:-}/$CKPT_STEP; ASSET=${ASSET_ID:-$(task_asset "$TASK")} ;;
  1a_rule)  REPO=$N0VTLA_1A; CONFIG=sim_dual_1a_rule
            CKPT=$N0VTLA_1A/checkpoints/sim_dual_1a_rule/1a_rulesim_$SHORT${EXP_SUFFIX:-}/$CKPT_STEP; ASSET=${ASSET_ID:-$(task_asset "$TASK")}
            export VTLA_1A_STATS=$N0VTLA_1A/assets/sim_dual_1a_rule/$ASSET/tactile_contact_stats.json ;;
  *) echo "unknown variant $VARIANT" >&2; exit 1 ;;
esac
TAG=${VARIANT}_${SHORT}${TAG_SUFFIX:-}
LOG_DIR=$SIM_RUNS/logs/eval; mkdir -p "$LOG_DIR"
SERVER_LOG=$LOG_DIR/${TAG}_server.log; CLIENT_LOG=$LOG_DIR/${TAG}_client.log
source /mnt/ssd1/guningquan/miniforge3/etc/profile.d/conda.sh

# --- policy server (vtla env) ---
set +u; conda activate vtla; set -u   # conda activate.d scripts read unset variables
( cd "$REPO" && export N0VTLA_DATA_HOME=$REPO/models HF_HOME=$REPO/models/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && \
  CUDA_VISIBLE_DEVICES=$SERVER_GPU PYTHONPATH=$REPO VTLA_ASSET_ID=$ASSET \
    exec python scripts/serve_zmq.py --config "$CONFIG" --ckpt "$CKPT" \
      --addr "tcp://*:$PORT" --default-prompt "$PROMPT" ) > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
  grep -q "N0-VTLA ZMQ server on" "$SERVER_LOG" && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "server died, see $SERVER_LOG"; exit 1; }
  sleep 5
done
grep "N0-VTLA ZMQ server on" "$SERVER_LOG" || { echo "server not ready, see $SERVER_LOG"; exit 1; }
set +u; conda deactivate; set -u

# --- simulator client (UniVTAC env) ---
set +u; conda activate UniVTAC; set -u   # conda activate.d scripts read unset variables
cd "$NEOSIM"
export CUDA_VISIBLE_DEVICES=$CLIENT_GPU UNIVTAC_HEADLESS=1 UNIVTAC_STEPS_PER_ACTION=$STEPS_PER_ACTION
export EVAL_TAG=$TAG OMNI_KIT_ACCEPT_EULA=YES VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_CLAMP_MAXALLOC=1   # see collect.sh: clamp layer for driver 595
echo "[eval] $TAG: task=$TASK config=$CONFIG ckpt=$CKPT seeds=$START_SEED+$TOTAL steps/action=$STEPS_PER_ACTION"
python eval/eval_n0vtla_dual_joint_cl.py "$TASK" "$TASK_CONFIG" \
  --server_port "$PORT" --prompt "$PROMPT" --exec_horizon 16 \
  --start_seed "$START_SEED" --total_num "$TOTAL" --headless --enable_cameras > "$CLIENT_LOG" 2>&1
tail -3 "$CLIENT_LOG"
