# NeoSim 仿真实验脚本（B0-sim 对比 1A-RULE-sim）

这些脚本把 N₀-VTLA 的仿真实验串成几条命令：采集 → 转换 → 训练 → 评测 → 统计。

- **适用环境：**本服务器，只用 GPU 2、3。
- **路径：**写死在 `scripts/tasks.sh` 里，换机器时先改这个文件。
- **在用的副本：**`/mnt/ssd2/guningquan/Programs_server/sim_runs/scripts/`；这里是它的一份拷贝，2026-10-26 冒烟规模跑通。

## 相关文档

- **NeoSim 的安装与使用：**fork `guningquan/N0-Foundation` 的 `n0vtla-sim` 分支：
  - `neosim/docs/QuickStart_Server_zh.md`：快速使用；
  - `neosim/docs/ServerSetupNotes.md`：安装问题与修复。
- **1A 的方法与实验设计：**研究文档《N₀-VTLA：仿真实验》（v1.1，第 10 节是跑通记录）。

## 前提

- conda 环境 `vtla`：训练、推理、转换。
- conda 环境 `UniVTAC`：仿真。
- NeoSim 位于 `N0-Foundation/neosim`。
- 驱动 595 的 Vulkan 补丁层已注册。脚本里已经设置了 `VK_CLAMP_MAXALLOC=1`。

## 脚本

| 脚本 | 用途 | 用法 |
|---|---|---|
| `tasks.sh` | 任务表：提示词、asset_id、各目录路径。其他脚本都会加载它 | — |
| `collect.sh` | 采集一个任务（`parallel_collect.sh`） | `collect.sh <task> [gpus=2,3] [workers=4] [config=collect_vtla]` |
| `collect_all.sh` | 依次采集 3 个任务 | `collect_all.sh [workers=4]` |
| `convert.sh` | HDF5 → LeRobot，再算归一化统计 | `convert.sh <task> [src_config=collect_vtla]` |
| `train.sh` | 训练 B0-sim 或 1A-RULE-sim（batch 32，GPU 2、3，phase A 2000 + 500） | `train.sh b0\|1a_rule <task> [额外的 train.sh 参数]` |
| `eval.sh` | 起一个服务端（`serve_zmq.py`）和一个仿真客户端，评测一个模型 | `eval.sh released\|b0\|1a_rule <task>` |
| `compare_eval.py` | 成功率、Wilson 置信区间、同种子上的精确 McNemar 检验 | `compare_eval.py A/results.jsonl B/results.jsonl --names A B` |
| `s0_phase*.sh` | 本服务器上实际执行过的安装步骤，依次为 1、2、2b、2c、2d | 环境已装好，仅供参考 |

**`train.sh` 的环境变量：**
- `DATA_ROOT`、`ASSET_ID`、`EXP_SUFFIX`；
- `PHASE_A_STEPS`、`PHASE_RAMP_STEPS`；
- `TRAIN_GPUS`、`BATCH`。

**`eval.sh` 的环境变量：**
- `SERVER_GPU`（默认 3）、`CLIENT_GPU`（默认 2）、`PORT`（默认 5557）；
- `TOTAL`（默认 100）、`START_SEED`（默认 1000000）；
- `STEPS_PER_ACTION`（默认 2）、`TASK_CONFIG`（默认 fasteval）；
- `CKPT_STEP`（默认 20000）、`EXP_SUFFIX`、`ASSET_ID`、`TAG_SUFFIX`；
- `EVAL_STEP_LIM`：由客户端直接读取。

注意：一个服务端只能接一个客户端，因为服务端每回合只保存一份触觉基准帧。并行评测时，每组要用不同的 `PORT`。

## 冒烟示例（跑通时用的命令）

```bash
# 训练 200 步（phase A 缩短为 50 + 20）
DATA_ROOT=/mnt/ssd2/guningquan/Dataset/neosim_vtla_smoke ASSET_ID=neosim_bowl_stack_smoke EXP_SUFFIX=_smoke \
PHASE_A_STEPS=50 PHASE_RAMP_STEPS=20 \
  bash scripts/train.sh b0 dual_bowl_place_stack --num-train-steps=200 --save-interval=200 --log-interval=10

# 在仿真里跑训练出的权重，1 个短回合
EXP_SUFFIX=_smoke ASSET_ID=neosim_bowl_stack_smoke CKPT_STEP=200 TOTAL=1 TAG_SUFFIX=_smoke PORT=5562 \
EVAL_STEP_LIM=400 bash scripts/eval.sh b0 dual_bowl_place_stack
```

## 正式实验前要先定的事

- **每个动作走 1 步还是 2 步：**
  - 做法：用发布的 checkpoint，按 `STEPS_PER_ACTION=1` 在种子 1000000–1000002 上再跑一次。按 2 步时是 3 次成功 2 次。
  - 判断：明显更接近 96%，就用 1 步；差不多，就用 2 步，并用 `EVAL_STEP_LIM` 放宽步数上限。
  - B0 和 1A 必须用同一个设置。
- **checkpoint 体积：**每个约 22.6 GB，训完要清理中间的 checkpoint。
