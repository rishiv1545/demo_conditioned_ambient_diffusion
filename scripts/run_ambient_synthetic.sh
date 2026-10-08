#!/usr/bin/env bash
# Mechanism study: does the ambient loss help when held-out-object demos are noisy? (small diffusion policy, env v2)
#   A  : sim demos, red/green tasks only (20/task)
#   B  : A + 24 noisy synthetic blue demos (naive)
#   C* : A + the same noisy demos with the ambient loss (t_min sweep)
#   O  : A + the same 24 blue demos without corruption (oracle)
# Re-runnable: finished runs (summary.json present) are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
OUT=outputs/ambient_synth
run() {  # name, blue-demo folder, extra train args
    local name=$1 blue=$2; shift 2
    [ -f $OUT/$name/summary.json ] && return 0
    $PY policy/train.py --run_name amb_$name --data data/sim_v2 $blue \
        --sim_tasks seen --sim_per_task 20 "$@" > $OUT/train_$name.log 2>&1
    $PY policy/evaluate.py --ckpt checkpoints/amb_$name/ckpt.pt --k 50 --out $OUT/$name --videos 1 > $OUT/eval_$name.log 2>&1
    grep -E "^(seen|heldout)" $OUT/eval_$name.log | sed "s/^/$name /"
}
mkdir -p $OUT
N=data/synthetic_v2; C=data/synthetic_clean_v2
run A   $N --sources sim
run B   $N --sources sim,synthetic --human_tasks heldout
run C50 $N --sources sim,synthetic --human_tasks heldout --ambient_t_min 50
run O   $C --sources sim,synthetic --human_tasks heldout
run C25 $N --sources sim,synthetic --human_tasks heldout --ambient_t_min 25
run C75 $N --sources sim,synthetic --human_tasks heldout --ambient_t_min 75
