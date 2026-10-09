#!/usr/bin/env bash
# Second small-policy sweep (after run_ambient_sweep.sh): is adding noisy demos on top of the clean ones better than
# discarding them, as a function of the amount of clean blue data? And the same with the REAL phone replays
# (data/human/3, retargeted + replayed in sim; the small policy is state-based, so they need no rendering).
#   K<n>          n clean demos per blue task, clean only (K4 = C and K24 = Ceil from the first sweep)
#   K<n>_N        + synthetic phone-like noisy demos (L2), naive;  K<n>_Na<t>: ambient
#   K<n>_P        + real phone replays (all 9 tasks, failed replays included), naive;  K<n>_Pa<t>: ambient
# Same recipe as run_ambient_sweep.sh (20 sim demos per red/green task, 20k steps, 50 episodes/task eval).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
OUT=outputs/ambient_sweep
N=data/synth_L2
P=data/human/3
mkdir -p $OUT

run() {  # name, extra train args
    local name=$1; shift
    [ -f $OUT/$name/summary.json ] && return 0
    $PY policy/train.py --run_name sweep_$name --data data/sim_v2 "$@" --sim_tasks seen --sim_per_task 20 \
        > $OUT/train_$name.log 2>&1
    $PY policy/evaluate.py --ckpt checkpoints/sweep_$name/ckpt.pt --k 50 --out $OUT/$name --videos 1 \
        > $OUT/eval_$name.log 2>&1
    echo "$name $(grep -E '^(seen|heldout)' $OUT/eval_$name.log | tr '\n' ' ')"
}

# 1) clean-data scaling with and without the noisy demos (synthetic, phone-like level)
for k in 24 8 12; do
    [ $k = 24 ] || run K$k --sources sim --clean_heldout_per_task $k
    run K${k}_N   $N --sources sim,synthetic --human_tasks all --clean_heldout_per_task $k
    run K${k}_Na25 $N --sources sim,synthetic --human_tasks all --clean_heldout_per_task $k --ambient_t_min 25
    run K${k}_Na50 $N --sources sim,synthetic --human_tasks all --clean_heldout_per_task $k --ambient_t_min 50
done
run K8_N0 $N --sources sim,synthetic --human_tasks heldout --clean_heldout_per_task 8   # noisy blue only

# 2) the real phone replays
for k in 4 24; do
    run K${k}_P    $P --sources sim,human --human_tasks all --clean_heldout_per_task $k
    run K${k}_Pa25 $P --sources sim,human --human_tasks all --clean_heldout_per_task $k --ambient_t_min 25
    run K${k}_Pa50 $P --sources sim,human --human_tasks all --clean_heldout_per_task $k --ambient_t_min 50
done
run K0_Pa25 $P --sources sim,human --human_tasks heldout --ambient_t_min 25   # phone blue only (N+amb analog)
