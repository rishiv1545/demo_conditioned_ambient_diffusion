"""Run one or more named runs (colab/runs.json) end to end on a Colab VM: data -> vision cache -> train -> probe -> eval.

Started detached by `colab/dcad.py run` (also usable from a notebook cell). Everything that matters lives on Drive, so
a disconnect or a new VM only costs the current stage, and every stage is idempotent:
- datasets: Drive dcad/datasets/<name>.tar -> /content/repo/data/lerobot/<name> (local disk; video decoding from Drive
  is slow). Upload a new one with `dcad.py push-data`.
- vision cache: Drive dcad/vision_cache/<name>_bf16 (trusted only with a COMPLETE marker), else built on the GPU.
- train: train_smolvla.py --resume into Drive dcad/checkpoints/vla/<run>.
- probe (all checkpoints) and eval (latest, --k eval_k) write into the run dir.
Status: <run dir>/job_status.json (stage, timings, error) and <run dir>/job.log; dcad/jobs/<job id>.json for the queue.

    python colab/job.py V1_phone_naive V2_phone_ambient_t03      # runs them one after the other
    python colab/job.py V0_2cam_unfrozen --stages probe,eval      # only some stages
"""
import argparse
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVE = "/content/drive/MyDrive/dcad"
OUT = f"{DRIVE}/checkpoints/vla"
STAGES = ["data", "cache", "train", "probe", "check", "eval", "grasp"]   # check: raw-vs-cached inference (vla/check_cache.py)
# grasp: grasp-offset bias/scatter and offline-vs-closed-loop error (vla/grasp_diag.py)


def load_runs():
    with open(os.path.join(REPO, "colab", "runs.json")) as f:
        return json.load(f)


ENV = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "USE_TF": "0", "PYTHONUNBUFFERED": "1"}


def sh(cmd, log):
    """Run a shell command from the repo root, streaming its output into the log; raise on failure."""
    log.write(f"\n$ {cmd}\n")
    log.flush()
    r = subprocess.run(cmd, shell=True, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                       env={**os.environ, **ENV})
    if r.returncode != 0:
        raise RuntimeError(f"exit {r.returncode}: {cmd}")


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def ensure_dataset(name, log):
    local = os.path.join(REPO, "data", "lerobot", name)
    if os.path.exists(os.path.join(local, "meta", "info.json")) and os.path.exists(os.path.join(local, "episodes.json")):
        log.write(f"dataset {name}: already on local disk\n")
        return local
    tar = f"{DRIVE}/datasets/{name}.tar"
    if os.path.exists(tar):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        sh(f"tar -xf {shlex.quote(tar)} -C {shlex.quote(os.path.dirname(local))}", log)
    elif os.environ.get("HF_TOKEN"):
        sh(f"hf download Rishi1545/dcad_{name} --repo-type dataset --local-dir {shlex.quote(local)} --quiet", log)
    else:
        raise FileNotFoundError(f"dataset {name}: no {tar} on Drive and no HF_TOKEN; run `dcad.py push-data` first")
    if not os.path.exists(os.path.join(local, "episodes.json")):
        raise FileNotFoundError(f"dataset {name}: episodes.json missing after download")
    return local


def ensure_cache(name, log):
    """Restore the frozen-vision cache from Drive, or build it here and save it to Drive."""
    local = os.path.join(REPO, "data", "lerobot", name)
    lc = os.path.join(local, "vision_cache")
    dc = f"{DRIVE}/vision_cache/{name}_bf16"
    if glob.glob(f"{lc}/*.npy") and os.path.exists(f"{lc}/LOCAL_COMPLETE"):
        log.write(f"cache {name}: already on local disk\n")
        return
    lock = f"{DRIVE}/vision_cache/{name}.building"   # another VM building the same cache: wait for it
    while not os.path.exists(f"{dc}/COMPLETE") and os.path.exists(lock) and time.time() - os.path.getmtime(lock) < 3 * 3600:
        log.write(f"cache {name}: another VM is building it, waiting\n")
        log.flush()
        time.sleep(120)
    if os.path.exists(f"{dc}/COMPLETE"):
        os.makedirs(lc, exist_ok=True)
        sh(f"cp {dc}/*.npy {dc}/*.json {lc}/", log)
    else:
        os.makedirs(os.path.dirname(lock), exist_ok=True)
        open(lock, "w").write(os.uname().nodename)
        sh(f"python vla/build_cache.py {shlex.quote(local)} --device cuda --batch 64", log)
        shutil.rmtree(dc, ignore_errors=True)
        os.makedirs(dc)
        sh(f"cp {lc}/*.npy {lc}/*.json {dc}/ && touch {dc}/COMPLETE", log)
        os.remove(lock)
    open(f"{lc}/LOCAL_COMPLETE", "w").close()


# early stopping on the blue probe (runs.json "early_stop": overrides): probe every saved checkpoint (save_every)
EARLY_STOP = {"cubes": "blue", "patience": 1000, "acc_tol": 2 / 32, "dist_tol_cm": 0.2, "max_nearest_cm": 3.5}
CLEAN_BLUE = "sim_blue_v2c_24"   # 24 clean sim demos per blue task; runs use the first clean_blue_per_task of each
PHONE = "phone_v2"


def dataset_specs(cfg):
    """[(dataset name, train_smolvla --data spec)] for a run: its "data" list, plus the clean blue demos
    ("clean_blue_per_task": N) and the phone replays ("phone": "all" | "blue")."""
    out = [(d.split("?")[0], f"data/lerobot/{d}") for d in cfg["data"]]   # "name?cube=blue" selects episodes
    n = cfg.get("clean_blue_per_task", 0)
    if n:
        out.append((CLEAN_BLUE, f"data/lerobot/{CLEAN_BLUE}?per_task={n}"))
    phone = cfg.get("phone")
    if phone:
        assert phone in ("all", "blue"), f"phone must be 'all' or 'blue', not {phone!r}"
        out.append((PHONE, f"data/lerobot/{PHONE}" + ("?cube=blue" if phone == "blue" else "")))
    return out


def latest_step(run_dir):
    steps = [int(os.path.basename(d)[5:]) for d in glob.glob(f"{run_dir}/step_*")
             if os.path.exists(f"{d}/trainable.pt")]
    return max(steps, default=0)


def probe_blue(run_dir):
    """{step: blue grounding accuracy} from the run's probe.txt (last entry per step)."""
    out, step = {}, None
    for line in open(f"{run_dir}/probe.txt"):
        m = re.match(r"step\s+(\d+):", line)
        if m:
            step = int(m.group(1))
        m = re.match(r"\s+blue\s+accuracy ([\d.]+)", line)
        if m and step is not None:
            out[step] = float(m.group(1))
    return out


def plateau_step(run_dir, after, tol=1 / 16):
    """Earliest checkpoint after `after` (the init step) whose blue probe accuracy is within tol (2 of 32 layouts)
    of the best one: where the blue probe plateaus."""
    acc = {s: v for s, v in probe_blue(run_dir).items() if s > after}
    assert acc, f"no probed checkpoints after step {after} in {run_dir}/probe.txt"
    best = max(acc.values())
    step = min(s for s, v in acc.items() if v >= best - tol)
    write_json(f"{run_dir}/plateau.json", {"step": step, "blue_acc": acc, "after": after, "tol": tol})
    return step


def probe_results(path):
    """{step: (accuracy over the queried cubes, median distance to the nearest cube in cm)} from a probe log."""
    out = {}
    if os.path.exists(path):
        for m in re.finditer(r"^step\s+(\d+): grounding accuracy ([\d.]+).*?to the nearest cube of any color ([\d.]+) cm",
                             open(path).read(), re.M):
            out[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return out


def plateaued(res, init_step, es):
    """True when neither the blue probe accuracy (by acc_tol) nor the nearest-cube distance (by dist_tol_cm) has
    improved on its running best for es["patience"] steps; the init checkpoint is the starting point."""
    steps = sorted(s for s in res if s >= init_step)
    if not steps or steps[-1] - init_step < es["patience"]:
        return False
    best_a, best_d, last = res[steps[0]][0], res[steps[0]][1], steps[0]
    for st in steps[1:]:
        acc, d = res[st]
        if acc >= best_a + es["acc_tol"]:
            best_a, last = acc, st
        if d <= best_d - es["dist_tol_cm"]:
            best_d, last = d, st
    return steps[-1] - last >= es["patience"]


def select_checkpoint(res, init_step, es):
    """Best blue probe accuracy among checkpoints after init whose nearest-cube distance is within max_nearest_cm
    (ties: smaller distance, then earlier); if none qualifies, the one with the smallest distance."""
    cand = {s: v for s, v in res.items() if s > init_step}
    ok = {s: v for s, v in cand.items() if v[1] <= es["max_nearest_cm"]}
    if ok:
        return min(ok, key=lambda s: (-ok[s][0], ok[s][1], s)), True
    return min(cand, key=lambda s: (cand[s][1], s)), False


def train_early_stop(run_dir, train_cmd, init_step, es, log):
    """Train while a watcher probes every saved checkpoint (blue only); write STOP once both probe metrics have
    plateaued (training saves and exits), pick the checkpoint, and delete the other checkpoints (588 MB each)."""
    plog, stop = f"{run_dir}/probe_es.txt", f"{run_dir}/STOP"
    if os.path.exists(stop):
        os.remove(stop)
    env = {**os.environ, **ENV}
    log.write(f"\n$ {train_cmd}\n")
    log.flush()
    tr = subprocess.Popen(train_cmd, shell=True, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=env)
    pcmd = (f"python vla/probe_grounding.py --run {run_dir} --watch --cubes {es['cubes']} --layouts 32 "
            f"--device cuda --log {plog}")
    pr = subprocess.Popen(pcmd, shell=True, cwd=REPO, stdout=open(f"{run_dir}/probe_es.out", "a"),
                          stderr=subprocess.STDOUT, env=env)
    while tr.poll() is None:
        time.sleep(20)
        if not os.path.exists(stop) and plateaued(probe_results(plog), init_step, es):
            open(stop, "w").close()
            log.write(f"plateau: STOP written at probed step {max(probe_results(plog))}\n")
    if tr.returncode != 0:
        pr.kill()
        raise RuntimeError(f"exit {tr.returncode}: {train_cmd}")
    last = latest_step(run_dir)
    while last not in probe_results(plog) and pr.poll() is None:   # let the watcher catch up
        if os.path.exists(stop) and plateaued(probe_results(plog), init_step, es):
            break      # already plateaued: the unprobed tail can't win on the plateau rule's terms
        time.sleep(20)
    pr.kill()
    res = probe_results(plog)
    step, passed = select_checkpoint(res, init_step, es)
    keep = {step, last}
    for st, d in [(int(os.path.basename(d)[5:]), d) for d in glob.glob(f"{run_dir}/step_*")]:
        if st not in keep:
            shutil.rmtree(d)
    write_json(f"{run_dir}/selected.json", {"step": step, "passed_distance_cut": passed, "last": last,
                                            "probe": {str(k): v for k, v in sorted(res.items())}, "rule": es})
    log.write(f"selected step {step} (distance cut passed: {passed}); kept {sorted(keep)}\n")
    return step


def init_from(src_name, run_dir, log):
    """Continue another run: copy its latest checkpoint (weights + optimizer state) and its loss.csv up to that step,
    so train_smolvla.py --resume picks it up and the source run stays untouched."""
    src = f"{OUT}/{src_name}"
    step = latest_step(src)
    if not step or not os.path.exists(f"{src}/step_{step:06d}/train_state.pt"):
        raise FileNotFoundError(f"init_from {src_name}: no checkpoint with train_state.pt in {src}")
    sh(f"cp -r {src}/step_{step:06d} {run_dir}/", log)
    if os.path.exists(f"{src}/loss.csv"):
        rows = open(f"{src}/loss.csv").read().splitlines()
        keep = [r for r in rows[1:] if r.split(",")[0].isdigit() and int(r.split(",")[0]) <= step]
        with open(f"{run_dir}/loss.csv", "w") as f:
            f.write("\n".join(rows[:1] + keep) + "\n")
    log.write(f"initialized from {src_name} step {step}\n")


def run_one(name, cfg, defaults, stages, job_id, eval_args="", eval_tag="", eval_steps=None):
    run_dir = f"{OUT}/{name}"
    os.makedirs(run_dir, exist_ok=True)
    status_path = f"{run_dir}/job_status{eval_tag}.json"   # tagged eval-only jobs don't clobber the main status
    status = {"run": name, "job": job_id, "stages": stages, "done": [], "stage": None, "error": None,
              "started": time.strftime("%Y-%m-%d %H:%M:%S"), "timings_s": {},
              "commit": subprocess.run("git rev-parse --short HEAD", shell=True, cwd=REPO,
                                       capture_output=True, text=True).stdout.strip()}
    specs = dataset_specs(cfg)
    train_args = f"{defaults['train']} {cfg.get('train', '')}"
    if cfg.get("ambient"):
        t = defaults.get("ambient_t_min")
        assert t, f"{name}: ambient run but defaults.ambient_t_min is not set (choose it from the small-policy sweep)"
        train_args += f" --ambient_t_min {t}"
    eval_k = cfg.get("eval_k", defaults.get("eval_k", 20))
    with open(f"{run_dir}/job.log", "a") as log:
        log.write(f"\n===== {status['started']} job {job_id} run {name} commit {status['commit']} stages {stages}\n")
        try:
            for st in stages:
                status["stage"] = st
                write_json(status_path, status)
                t0 = time.time()
                if st == "data":
                    for d, _ in specs:
                        ensure_dataset(d, log)
                elif st == "cache":
                    if "--features raw" not in train_args:
                        for d, _ in specs:
                            ensure_cache(d, log)
                elif st == "train":
                    if cfg.get("init_from") and not latest_step(run_dir):
                        init_from(cfg["init_from"], run_dir, log)
                    if cfg.get("init_from") and "--stats_from" not in train_args:   # keep the source's normalization
                        train_args += f" --stats_from {OUT}/{cfg['init_from']}"
                    cmd = (f"python vla/train_smolvla.py --run_name {name} --data "
                           f"{' '.join(shlex.quote(sp) for _, sp in specs)} --out {OUT} {train_args}")
                    if cfg.get("early_stop"):
                        es = {**EARLY_STOP, **cfg["early_stop"]}
                        init_step = latest_step(f"{OUT}/{cfg['init_from']}") if cfg.get("init_from") else 0
                        if not os.path.exists(f"{run_dir}/selected.json"):
                            train_early_stop(run_dir, cmd, init_step, es, log)
                    else:
                        sh(cmd, log)
                elif st == "probe":
                    probe_args = cfg.get("probe", defaults.get("probe", ""))
                    pstep = cfg.get("probe_step", "all")
                    if pstep == "selected":   # full probe (all cubes) of the early-stopping checkpoint
                        pstep = json.load(open(f"{run_dir}/selected.json"))["step"]
                    sh(f"python vla/probe_grounding.py --run {run_dir} --step {pstep} --device cuda {probe_args} "
                       f"--log {run_dir}/probe.txt", log)
                elif st == "check":
                    step = (eval_steps or [latest_step(run_dir)])[-1]
                    sh(f"python vla/check_cache.py --run {run_dir} --step {step} --n 200 --device cuda "
                       f"--out {run_dir}/check_cache_step{step:06d}.json", log)
                elif st == "grasp":
                    step = (eval_steps or [latest_step(run_dir)])[-1]
                    sh(f"python vla/grasp_diag.py --run {run_dir} --step {step} --device cuda "
                       f"--out {run_dir}/grasp_diag_step{step:06d}.json", log)
                elif st == "eval" and eval_k > 0:
                    if cfg.get("eval_steps") == "selected" and not eval_steps:   # early-stopping checkpoint
                        eval_steps = [json.load(open(f"{run_dir}/selected.json"))["step"]]
                    if cfg.get("eval_steps") == "plateau" and not eval_steps:   # the blue probe's plateau checkpoint
                        init_step = latest_step(f"{OUT}/{cfg['init_from']}") if cfg.get("init_from") else 0
                        eval_steps = [plateau_step(run_dir, init_step)]
                        log.write(f"plateau step {eval_steps[0]}\n")
                    # variants: [[tag, args], ...] from runs.json (e.g. normal and zero-noise sampling), else the CLI's
                    variants = cfg.get("eval_variants") or [[eval_tag, eval_args]]
                    for step in eval_steps or cfg.get("eval_steps") or [latest_step(run_dir)]:   # e.g. a step matching another run
                        for tag, vargs in variants:
                            ed = f"{run_dir}/eval_step{step:06d}_k{eval_k}{tag}"
                            if os.path.exists(f"{ed}/summary.json"):
                                continue
                            sh(f"python vla/eval_smolvla.py --run {run_dir} --step {step} --k {eval_k} "
                               f"--out {ed} --device cuda {cfg.get('eval_args', '')} {vargs}", log)
                status["timings_s"][st] = round(time.time() - t0)
                status["done"].append(st)
            status["stage"] = "finished"
        except Exception as e:  # keep going with the next run in the queue; the error is in the status file
            status["error"] = f"{type(e).__name__}: {e}"
            log.write(traceback.format_exc())
        status["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json(status_path, status)
    return status


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--stages", default=",".join(STAGES))
    p.add_argument("--job_id", default=time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--eval_args", default="", help="extra eval_smolvla.py args, e.g. '--n_action_steps 5'")
    p.add_argument("--eval_tag", default="", help="suffix for eval dirs and the status file, e.g. _nas5")
    p.add_argument("--eval_steps", default="", help="comma list of checkpoints to evaluate/check (default: per run)")
    a = p.parse_args()
    stages = [s for s in a.stages.split(",") if s]
    if any(st in stages for st in ("cache", "train", "probe", "check", "eval", "grasp")) and "data" not in stages:
        stages = ["data"] + stages   # every later stage needs the datasets (episodes.json) on this VM
    assert set(stages) <= set(STAGES), f"stages must be in {STAGES}"
    assert os.path.isdir("/content/drive/MyDrive"), "Drive is not mounted (run `colab drivemount -s <session>`)"
    reg = load_runs()
    missing = [r for r in a.runs if r not in reg["runs"]]
    assert not missing, f"unknown runs {missing}; add them to colab/runs.json"
    os.makedirs(f"{DRIVE}/jobs", exist_ok=True)
    job_path = f"{DRIVE}/jobs/{a.job_id}.json"
    job = {"job": a.job_id, "runs": a.runs, "stages": stages, "pid": os.getpid(), "state": "running", "results": {},
           "eval_tag": a.eval_tag}
    write_json(job_path, job)
    for r in a.runs:
        job["current"] = r
        write_json(job_path, job)
        steps = [int(x) for x in a.eval_steps.split(",") if x] or None
        s = run_one(r, reg["runs"][r], reg["defaults"], stages, a.job_id, a.eval_args, a.eval_tag, steps)
        job["results"][r] = {"error": s["error"], "done": s["done"]}
    job["current"] = None
    job["state"] = "failed" if any(v["error"] for v in job["results"].values()) else "finished"
    write_json(job_path, job)
    print(json.dumps(job, indent=2))
    sys.exit(1 if job["state"] == "failed" else 0)


if __name__ == "__main__":
    main()
