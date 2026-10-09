#!/usr/bin/env bash
# Mechanism study on the small diffusion policy (Mac, MPS; env v2): can noisy demos improve post-training when clean
# data is scarce, and does the ambient loss make them usable? Same structure as the VLA runs (PHASE2_PLAN.md):
#   C          red/green sim (20/task) + NCLEAN clean sim demos per blue task
#   Ceil       red/green sim + 24 clean demos per blue task
#   L<k>_CN    C + noisy demos for all tasks (4 per red/green task, 8 per blue task: the phone-data counts), naive
#   L<k>_CNa<t> same, ambient loss with t_min = t (of T = 100)
#   L<k>_Na<t> red/green sim + noisy blue demos only (no clean blue), ambient
# Noise levels (synthetic, phone-like: per-episode xy/z offset, jitter, gripper timing; scripts/make_synthetic_noisy.py):
#   L1 = 0.4x, L2 = 1x (xy sigma 2.5 cm, the measured phone grasp error), L3 = 2x.
# Each run: 20k steps (~18 min on MPS) + 50-episode/task eval. Re-runnable: finished runs are skipped.
# Results: outputs/ambient_sweep/<run>/summary.json; table + t_min choice: python scripts/summarize_ambient_sweep.py
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
OUT=outputs/ambient_sweep
NCLEAN=${NCLEAN:-4}
mkdir -p $OUT

noisy() {  # level name, scale: corrupted copies of sim_v2 demos 150+ (disjoint from the clean ones used for training)
    local d=data/synth_$1 f=$2
    [ -f $d/COMPLETE ] && return 0
    rm -rf $d
    local args="--src data/sim_v2 --out $d --xy_sigma $(echo "0.025*$f" | bc -l) --z_sigma $(echo "0.01*$f" | bc -l)
                --jitter $(echo "0.003*$f" | bc -l) --grip_shift $(printf '%.0f' "$(echo "3*$f" | bc -l)")"
    $PY scripts/make_synthetic_noisy.py $args --tasks seen --per_task 4 --seed 1 > $OUT/noisy_$1.log
    $PY scripts/make_synthetic_noisy.py $args --tasks heldout --per_task 8 --seed 2 >> $OUT/noisy_$1.log
    touch $d/COMPLETE
}

run() {  # name, extra train args
    local name=$1; shift
    [ -f $OUT/$name/summary.json ] && return 0
    $PY policy/train.py --run_name sweep_$name --data data/sim_v2 "$@" --sim_tasks seen --sim_per_task 20 \
        > $OUT/train_$name.log 2>&1
    $PY policy/evaluate.py --ckpt checkpoints/sweep_$name/ckpt.pt --k 50 --out $OUT/$name --videos 1 \
        > $OUT/eval_$name.log 2>&1
    echo "$name $(grep -E '^(seen|heldout)' $OUT/eval_$name.log | tr '\n' ' ')"
}

for L in "L2 1.0" "L1 0.4" "L3 2.0"; do noisy $L; done

run C    --sources sim --clean_heldout_per_task $NCLEAN
run Ceil --sources sim --clean_heldout_per_task 24
for L in L2 L1 L3; do
    N=data/synth_$L
    run ${L}_CN    $N --sources sim,synthetic --human_tasks all --clean_heldout_per_task $NCLEAN
    for t in 25 50 10 75; do
        run ${L}_CNa$t $N --sources sim,synthetic --human_tasks all --clean_heldout_per_task $NCLEAN --ambient_t_min $t
    done
    for t in 25 50 75; do
        run ${L}_Na$t  $N --sources sim,synthetic --human_tasks heldout --ambient_t_min $t
    done
done
