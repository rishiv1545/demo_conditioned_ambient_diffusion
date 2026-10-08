#!/usr/bin/env bash
# Overnight V0 (2026-10-08, revised): env v2, scene + wrist cameras at 512 px, 100 sim demos per red/green task
# (600), 15k steps at constant lr 1e-4 (300-step warmup), effective batch 32, checkpoints every 2500 steps;
# then the grounding/localization probe on every checkpoint, then a 20-episode/task eval of the last checkpoint.
# One MPS job at a time. Re-runnable (export/cache/training resume or are skipped).
#   caffeinate -i bash scripts/run_v0_2cam_mac.sh 2>&1 | tee -a outputs/vla/v0_2cam.log
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
DATA=data/lerobot/sim_seen_v2c_100
RUN=V0_2cam
[ -f $DATA/episodes.json ] || $PY vla/export_lerobot.py --src data/sim_v2 --sources sim --tasks seen --per_task 100 \
    --name sim_seen_v2c_100
$PY vla/train_smolvla.py --run_name $RUN --data $DATA --out checkpoints/vla --steps 15000 --lr 1e-4 \
    --lr_schedule constant --warmup 300 --batch 16 --grad_accum 2 --save_every 2500 --resume
$PY vla/probe_grounding.py --run checkpoints/vla/$RUN --step all | tee outputs/vla/$RUN.probe.txt
[ -f outputs/vla/$RUN/final/summary.json ] || \
    $PY vla/eval_smolvla.py --run checkpoints/vla/$RUN --step latest --k 20 --out outputs/vla/$RUN/final
