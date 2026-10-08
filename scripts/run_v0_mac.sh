#!/usr/bin/env bash
# V0 (sim only) on the Mac: train from the vision-feature cache, evaluate checkpoints, final eval on the best.
# Safe to re-run: training resumes from the latest checkpoint and finished evaluations are skipped.
#   caffeinate -i bash scripts/run_v0_mac.sh 2>&1 | tee -a outputs/vla/V0_mac.log
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
RUN=V0
STEPS=${STEPS:-5000}
OUT=checkpoints/vla
EVAL=outputs/vla/$RUN

$PY vla/train_smolvla.py --run_name $RUN --data data/lerobot/sim_seen_v2 --out $OUT --steps $STEPS \
    --batch 16 --grad_accum 2 --save_every 500 --resume

# intermediate evals: 20 episodes/task every 1000 steps
for s in $(seq 1000 1000 $STEPS); do
    d=$EVAL/step_$s
    [ -f $d/summary.json ] || $PY vla/eval_smolvla.py --run $OUT/$RUN --step $s --k 20 --out $d
done

# final: 50 episodes/task on the checkpoint with the best seen-task success
best=$($PY - <<EOF
import glob, json, re
r = [(json.load(open(f))["seen"]["success"], int(re.search(r"step_(\d+)", f).group(1)))
     for f in glob.glob("$EVAL/step_*/summary.json")]
print(max(r)[1])
EOF
)
echo "best V0 checkpoint by seen-task success: step $best"
[ -f $EVAL/final_step_$best/summary.json ] || \
    $PY vla/eval_smolvla.py --run $OUT/$RUN --step $best --k 50 --videos 3 --out $EVAL/final_step_$best
