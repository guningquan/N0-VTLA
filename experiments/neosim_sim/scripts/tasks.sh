# Task table for the N₀-VTLA sim experiments (N₀-VTLA：仿真实验.md, S4 table). Source this file.
ROOT=/mnt/ssd2/guningquan/Programs_server
RAW_ROOT=/mnt/ssd2/guningquan/Dataset/neosim_raw
LEROBOT_ROOT=/mnt/ssd2/guningquan/Dataset/neosim_vtla
N0VTLA=$ROOT/N0-VTLA
N0VTLA_1A=$ROOT/N0-VTLA-1A
NEOSIM=$ROOT/N0-Foundation/neosim
SIM_RUNS=$ROOT/sim_runs
BASE_CKPT=$N0VTLA/checkpoints/n0-vtla-base

task_prompt() {
  case "$1" in
    dual_bowl_place_stack)   echo "Use both arms to place and stack the bowls" ;;
    dual_cup_handover_place) echo "Use both arms to hand over the cup and place it" ;;
    dual_screw_sleeve)       echo "Use both arms to insert the screw into the sleeve" ;;
    dual_gear_holder)        echo "Use both arms to place the gears on the holder" ;;
    *) echo "unknown task $1" >&2; return 1 ;;
  esac
}
task_asset() {
  case "$1" in
    dual_bowl_place_stack)   echo neosim_bowl_stack ;;
    dual_cup_handover_place) echo neosim_cup_handover ;;
    dual_screw_sleeve)       echo neosim_insert_screw ;;
    dual_gear_holder)        echo neosim_place_gears ;;
    *) echo "unknown task $1" >&2; return 1 ;;
  esac
}
# short name used in experiment names: neosim_cup_handover -> cup_handover
task_short() { task_asset "$1" | sed 's/^neosim_//'; }
