#!/usr/bin/env bash
# Overnight "option 2" (2026-10-08): V0 on env v2 with 4x data (200 sim demos per red/green task), 15k steps,
# lr held at 1e-4 after warmup, checkpoints every 2500 steps, grounding probe on every checkpoint after training,
# then a 20-episode/task eval of the last checkpoint, then the remaining ambient-synthetic runs.
# One GPU (MPS) job at a time. Re-runnable: export/cache/training/evals resume or skip.
#   caffeinate -i bash scripts/run_option2_mac.sh 2>&1 | tee -a outputs/vla/option2.log
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
DATA=data/lerobot/sim_seen_v2_200
RUN=V0_x4_const
[ -f $DATA/episodes.json ] || $PY vla/export_lerobot.py --src data/sim_v2 --sources sim --tasks seen --per_task 200 \
    --name sim_seen_v2_200
$PY vla/train_smolvla.py --run_name $RUN --data $DATA --out checkpoints/vla --steps 15000 --lr 1e-4 \
    --lr_schedule constant --warmup 300 --batch 16 --grad_accum 2 --save_every 2500 --resume
$PY vla/probe_grounding.py --run checkpoints/vla/$RUN --step all | tee outputs/vla/$RUN.probe.txt
[ -f outputs/vla/$RUN/final/summary.json ] || \
    $PY vla/eval_smolvla.py --run checkpoints/vla/$RUN --step latest --k 20 --out outputs/vla/$RUN/final
bash scripts/run_ambient_synthetic.sh
