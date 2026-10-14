"""Self-improvement loop for a trained single-task run: roll out in sim, keep the successes as clean data, retrain.

Each round r (on the VM, from colab/job.py stage "self_improve"):
  1. N rollouts of the latest checkpoint on fresh layouts (seeds 2_000_000 + r * 100_000 + task * 10_000 + k, never
     the eval seeds), default sampling noise for diversity; every rollout is saved (eval_smolvla.py --save_episodes).
  2. The successful rollouts of this round are exported as their own LeRobot dataset (<run>_roll_r<r>, sigma_n 0,
     i.e. clean; with --episode_info unlisted sim episodes get t_min 0 and quality +1), so earlier rounds are
     never re-rendered.
  3. Training continues the same run dir for --steps_per_round steps (fresh cosine from the resume step, warmup
     --resume_warmup) on the original data plus all rollout datasets so far, keeping the run's normalization.
  4. Zero-noise eval (--eval_k episodes) of the new checkpoint.
A summary per round goes to <run_dir>/self_improve.json.

    python vla/self_improve.py --run_dir <OUT>/T1_x --data 'data/lerobot/sim_bo_320?per_task=100' --task blue-orange \
        --rounds 3 --n 100 --steps_per_round 3000 --train_args "--unfreeze_vlm 1 --vlm_lr 1e-5"
"""
import argparse
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from vla.common import list_checkpoints  # noqa: E402


def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=REPO)


def eval_summary(d):
    s = json.load(open(os.path.join(d, "summary.json")))
    per = s["per_task"]
    return {"success": sum(per.values()) / len(per), "outcomes": next(iter(s["outcomes"].values()), {})}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data", nargs="+", required=True, help="the run's original training data specs")
    p.add_argument("--task", default="blue-orange")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--n", type=int, default=100, help="rollouts per round")
    p.add_argument("--steps_per_round", type=int, default=3000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr_floor", type=float, default=2.5e-6)
    p.add_argument("--resume_warmup", type=int, default=200)
    p.add_argument("--train_args", default="--unfreeze_vlm 1 --vlm_lr 1e-5",
                   help="the run's other train args (VLM, episode_info, ambient, quality tag), kept every round")
    p.add_argument("--eval_k", type=int, default=20)
    p.add_argument("--eval_tag", default="_zn", help="eval dir suffix (zero-noise eval of each round)")
    p.add_argument("--rollout_noise", type=float, default=1.0)
    a = p.parse_args()
    run = os.path.basename(os.path.normpath(a.run_dir))
    out = os.path.dirname(os.path.normpath(a.run_dir))
    summ_path = os.path.join(a.run_dir, "self_improve.json")
    summ = json.load(open(summ_path)) if os.path.exists(summ_path) else {"rounds": []}
    roll_data = [f"data/lerobot/{r['dataset']}" for r in summ["rounds"] if r.get("dataset")]
    for r in range(len(summ["rounds"]), a.rounds):
        t0 = time.time()
        step = list_checkpoints(a.run_dir)[-1][0]
        rec = {"round": r, "from_step": step}
        # 1. rollouts
        ep_dir = os.path.join(REPO, "data", "rollouts", run, f"r{r}")
        shutil.rmtree(ep_dir, ignore_errors=True)
        sh(f"python vla/eval_smolvla.py --run {a.run_dir} --step {step} --tasks {a.task} --k {a.n} "
           f"--seed_base {2_000_000 + r * 100_000} --noise_scale {a.rollout_noise} --videos 0 --save_episodes {ep_dir} "
           f"--out {a.run_dir}/rollouts_r{r} --device cuda")
        files = sorted(glob.glob(os.path.join(ep_dir, "*.npz")))
        ok = [f for f in files if bool(np.load(f, allow_pickle=True)["success"])]
        rec.update(rollouts=len(files), successes=len(ok), rollout_success=len(ok) / max(1, len(files)))
        rec["t_rollouts_s"] = round(time.time() - t0)
        # 2. export this round's successes
        if ok:
            succ = os.path.join(ep_dir, "success")
            os.makedirs(succ, exist_ok=True)
            for f in ok:
                shutil.copy(f, succ)
            name = f"{run}_roll_r{r}"
            sh(f"python vla/export_lerobot.py --src {succ} --sources sim --tasks heldout --heldout {a.task} "
               f"--per_task {len(ok)} --name {name}")
            roll_data.append(f"data/lerobot/{name}")
            rec["dataset"] = name
        rec["t_export_s"] = round(time.time() - t0) - rec["t_rollouts_s"]
        # 3. continue training on original + all rollout data
        end = step + a.steps_per_round
        data = " ".join(shlex.quote(d) for d in a.data + roll_data)
        sh(f"python vla/train_smolvla.py --run_name {run} --data {data} --out {out} --resume --device cuda "
           f"--batch 32 --steps {end} --lr {a.lr} --lr_floor {a.lr_floor} --lr_schedule cosine_from_resume "
           f"--resume_warmup {a.resume_warmup} --save_from {step} --save_every {a.steps_per_round} "
           f"--stats_from {a.run_dir} {a.train_args}")
        # 4. zero-noise eval
        ed = f"{a.run_dir}/eval_step{end:06d}_k{a.eval_k}{a.eval_tag}"
        sh(f"python vla/eval_smolvla.py --run {a.run_dir} --step {end} --tasks {a.task} --k {a.eval_k} "
           f"--noise_scale 0 --out {ed} --device cuda")
        rec.update(to_step=end, eval=eval_summary(ed), minutes=round((time.time() - t0) / 60, 1))
        summ["rounds"].append(rec)
        json.dump(summ, open(summ_path, "w"), indent=1)
        print(f"round {r}: rollouts {rec['successes']}/{rec['rollouts']} ok, eval {rec['eval']['success']:.0%} "
              f"at step {end}, {rec['minutes']} min", flush=True)


if __name__ == "__main__":
    main()
