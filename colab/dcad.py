"""Drive Colab GPU runs from the Mac with the colab CLI (`uv tool install google-colab-cli`; `colab sessions` to log in).

    python colab/dcad.py up                          # A100 session 'dcad' + code at the local HEAD (must be pushed)
    python colab/dcad.py push-data data/lerobot/phone_v2   # tar a dataset (no cache) -> Drive dcad/datasets/
    python colab/dcad.py run V1_phone_naive V2_phone_ambient_t03   # queue runs (colab/runs.json), detached
    python colab/dcad.py status                      # latest job: stage, log tail, loss, probe, GPU
    python colab/dcad.py wait --stop                 # poll until the job ends, fetch results, stop the VM
    python colab/dcad.py fetch V1_phone_naive        # small results -> outputs/colab/<run>/
    python colab/dcad.py down                        # stop the VM (billing stops)

One manual step per new VM: `colab drivemount -s dcad` (interactive). Jobs run with nohup on the VM and keep all
state on Drive (see colab/job.py), so the Mac can sleep or disconnect. A running VM bills (~5 units/h on an A100)
until `down` or `wait --stop`, even when idle.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

COLAB = os.path.expanduser("~/.local/bin/colab")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE = "/content/repo"
GIT_URL = "https://github.com/rishiv1545/demo_conditioned_ambient_diffusion.git"
DRIVE = "/content/drive/MyDrive/dcad"
OUT = f"{DRIVE}/checkpoints/vla"


def colab(*args, check=True, capture=True):
    r = subprocess.run([COLAB, *args], stdin=subprocess.DEVNULL, capture_output=capture, text=True)
    if check and r.returncode != 0:
        sys.exit(f"colab {' '.join(args)} failed:\n{(r.stdout or '') + (r.stderr or '')}")
    return r.stdout if capture else ""


def remote(code, session, timeout=120):
    """Run Python code in the VM's kernel and return its stdout. `colab exec` exits 0 even when the code raises (the
    traceback goes to stderr), so a traceback on stderr is turned into a failure here."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
    try:
        r = subprocess.run([COLAB, "exec", "-s", session, "--timeout", str(timeout), "-f", f.name],
                           stdin=subprocess.DEVNULL, capture_output=True, text=True)
    finally:
        os.remove(f.name)
    err = re.sub(r"\x1b\[[0-9;]*m", "", r.stderr)
    if r.returncode != 0 or "Traceback" in err:
        sys.exit(f"remote code failed:\n{r.stdout}\n{err[-3000:]}")
    return r.stdout


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def pushed_head():
    """The local HEAD, which the VM checks out. Refuses if it isn't on GitHub (the VM couldn't fetch it)."""
    if git("status", "--porcelain", "--untracked-files=no"):
        print("warning: uncommitted changes are NOT on the VM (only committed + pushed code is)")
    head = git("rev-parse", "HEAD")
    subprocess.run(["git", "fetch", "-q", "origin"], cwd=REPO)
    if not git("branch", "-r", "--contains", head):
        sys.exit(f"HEAD {head[:7]} is not pushed; `git push` first")
    return head


def session_exists(session):
    """Whether the named session is running. Exits (never answers False) if the session list can't be read, e.g.
    a network blip: answering False would make `up` create a second, billed VM."""
    r = subprocess.run([COLAB, "sessions"], stdin=subprocess.DEVNULL, capture_output=True, text=True)
    out = r.stdout
    if r.returncode != 0 or "Traceback" in r.stderr or "authorize" in (r.stdout + r.stderr):
        sys.exit(f"could not list Colab sessions (network or login problem); not creating one:\n{r.stderr[-500:]}")
    return any(line.startswith(f"[{session}] ") for line in out.splitlines())   # "[dcad] gpu-a100-... | Hardware: ..."


def cmd_up(a):
    if not session_exists(a.session):
        print(f"creating session {a.session} ({a.gpu}) ...")
        colab("new", "-s", a.session, "--gpu", a.gpu, capture=False)
    head = pushed_head()
    out = remote(f"""
import os, subprocess, hashlib
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True)
if not os.path.exists("{REMOTE}/.git"):
    print(sh("git clone -q {GIT_URL} {REMOTE}").stderr)
r = sh("cd {REMOTE} && git fetch -q origin && git checkout -q -f --detach {head} && git log --oneline -1")
print("code:", r.stdout.strip(), r.stderr.strip())
req = hashlib.md5(open("{REMOTE}/requirements.txt", "rb").read()).hexdigest()
mark = "/content/.dcad_installed"
if not os.path.exists(mark) or open(mark).read() != req:
    r = sh("cd {REMOTE} && pip install -q -r requirements.txt")
    print("pip install:", "ok" if r.returncode == 0 else r.stdout[-2000:] + r.stderr[-2000:])
    if r.returncode == 0: open(mark, "w").write(req)
else:
    print("requirements: already installed")
print("gpu:", sh("nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader").stdout.strip())
print("drive:", "mounted" if os.path.isdir("/content/drive/MyDrive") else "NOT MOUNTED")
""", a.session, timeout=1800)
    print(out)
    if "NOT MOUNTED" in out:
        print(f"\n>>> mount Drive (interactive): {COLAB} drivemount -s {a.session}")
        return 2
    return 0


def cmd_push_data(a):
    src = os.path.abspath(a.path)
    name = a.name or os.path.basename(src.rstrip("/"))
    assert os.path.exists(os.path.join(src, "episodes.json")), f"{src}/episodes.json missing (not an exported dataset?)"
    if not a.no_sync and cmd_up(a):   # a stopped VM is recreated; Drive must be mounted to store the tar
        return 2
    with tempfile.TemporaryDirectory() as td:
        tar = os.path.join(td, f"{name}.tar")
        subprocess.run(["tar", "-cf", tar, "--exclude", "vision_cache", "-C", os.path.dirname(src),
                        "-s", f"|^{os.path.basename(src)}|{name}|", os.path.basename(src)], check=True)
        # the Jupyter contents API rejects large single uploads (a 456 MB file got HTTP 400): send 64 MB parts
        subprocess.run(["split", "-b", "64m", tar, os.path.join(td, "part_")], check=True)
        os.remove(tar)
        parts = sorted(f for f in os.listdir(td) if f.startswith("part_"))
        remote(f"import os, shutil; shutil.rmtree('/content/upload_{name}', True); os.makedirs('/content/upload_{name}')",
               a.session)
        for i, part in enumerate(parts):
            print(f"uploading part {i + 1}/{len(parts)}", flush=True)
            colab("upload", "-s", a.session, os.path.join(td, part), f"/content/upload_{name}/{part}")
    print(remote(f"""
import os, shutil, subprocess
subprocess.run("cat /content/upload_{name}/part_* > /content/{name}.tar && rm -r /content/upload_{name}", shell=True, check=True)
os.makedirs("{DRIVE}/datasets", exist_ok=True)
if os.path.exists("{DRIVE}/datasets/{name}.tar"):   # replacing a dataset invalidates its cache (a first upload of
    shutil.rmtree("{DRIVE}/vision_cache/{name}_bf16", ignore_errors=True)   # a Hub dataset keeps the existing one)
    shutil.rmtree("{REMOTE}/data/lerobot/{name}", ignore_errors=True)
shutil.copy("/content/{name}.tar", "{DRIVE}/datasets/{name}.tar.tmp")
os.replace("{DRIVE}/datasets/{name}.tar.tmp", "{DRIVE}/datasets/{name}.tar")
os.remove("/content/{name}.tar")
print("on Drive:", "{DRIVE}/datasets/{name}.tar", os.path.getsize("{DRIVE}/datasets/{name}.tar") // 2**20, "MB")
""", a.session, timeout=600))


def cmd_run(a):
    with open(os.path.join(REPO, "colab", "runs.json")) as f:
        known = json.load(f)["runs"]
    missing = [r for r in a.runs if r not in known]
    if missing:
        sys.exit(f"unknown runs {missing}; add them to colab/runs.json (and push)")
    if cmd_up(a):
        return 2
    job_id = time.strftime("%Y%m%d-%H%M%S")
    env = {"HF_TOKEN": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}
    print(remote(f"""
import subprocess, os, glob
def argv(p):
    try: return open(p, "rb").read().split(b"\\0")
    except OSError: return []
busy = [p.split("/")[2] for p in glob.glob("/proc/[0-9]*/cmdline") if b"colab/job.py" in argv(p)[1:2]]
assert not busy, f"a job is already running on this VM (pids {{busy}}); one GPU job at a time"
log = open("/content/job_{job_id}.out", "w")
p = subprocess.Popen(["nohup", "python", "colab/job.py", *{a.runs!r}, "--stages", {a.stages!r}, "--job_id", "{job_id}"],
                     cwd="{REMOTE}", stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                     env={{**os.environ, **{env!r}}})
print("started job {job_id} pid", p.pid, "runs", {a.runs!r})
""", a.session))
    print(f"follow with: python colab/dcad.py status   |   python colab/dcad.py wait --stop")


STATUS_CODE = f"""
import glob, json, os, subprocess
jobs = sorted(glob.glob("{DRIVE}/jobs/*.json"))
want = {{JOB!r}}
path = f"{DRIVE}/jobs/{{want}}.json" if want else (jobs[-1] if jobs else None)
if not path or not os.path.exists(path):
    print("no jobs"); raise SystemExit
job = json.load(open(path))
def alive(pid):   # the kernel never reaps the detached job, so a finished one lingers as a zombie
    try: return open(f"/proc/{{pid}}/stat").read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError: return False
alive = alive(job["pid"]) and job["state"] == "running"
if job["state"] == "running" and not alive:
    job["state"] = "died"   # VM restarted or the process was killed
print("JOB", job["job"], "state", job["state"], "runs", job["runs"], "current", job.get("current"))
for r in job["runs"]:
    d = f"{OUT}/{{r}}"
    s = json.load(open(f"{{d}}/job_status.json")) if os.path.exists(f"{{d}}/job_status.json") else {{}}
    if s.get("job") != job["job"]:
        print(f"-- {{r}}: not started"); continue
    print(f"-- {{r}}: stage {{s['stage']}} done {{s['done']}} timings {{s['timings_s']}} error {{s['error']}}")
    if os.path.exists(f"{{d}}/loss.csv"):
        rows = open(f"{{d}}/loss.csv").read().split()
        print("   loss.csv last:", rows[-1] if len(rows) > 1 else "(empty)")
    if os.path.exists(f"{{d}}/probe.txt"):
        print("   probe:\\n     " + "\\n     ".join(open(f"{{d}}/probe.txt").read().strip().splitlines()[-{{N}}:]))
    if r == job.get("current") and os.path.exists(f"{{d}}/job.log"):
        tail = open(f"{{d}}/job.log", errors="replace").read().replace("\\r", "\\n").strip().splitlines()[-{{N}}:]
        print("   log:\\n     " + "\\n     ".join(tail))
print("gpu:", subprocess.run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
      shell=True, capture_output=True, text=True).stdout.strip())
"""


def status(a, n=12):
    return remote(STATUS_CODE.replace("{JOB!r}", repr(a.job)).replace("{N}", str(n)), a.session)


def cmd_status(a):
    print(status(a, a.lines))


def cmd_fetch(a):
    for run in a.runs:
        dst = os.path.join(REPO, "outputs", "colab", run)
        os.makedirs(dst, exist_ok=True)
        files = remote(f"""
import glob, os
d = "{OUT}/{run}"
fs = [f for f in ["run.json", "loss.csv", "probe.txt", "job_status.json", "job.log"] if os.path.exists(f"{{d}}/{{f}}")]
fs += [os.path.relpath(p, d) for p in glob.glob(f"{{d}}/eval_*/summary.json") + glob.glob(f"{{d}}/eval_*/episodes.csv")]
print("\\n".join(fs))
""", a.session).split()
        for f in files:
            os.makedirs(os.path.dirname(os.path.join(dst, f)), exist_ok=True)
            colab("download", "-s", a.session, f"{OUT}/{run}/{f}", os.path.join(dst, f))
        print(f"{run}: {len(files)} files -> {os.path.relpath(dst, REPO)}")


def cmd_wait(a):
    fails = 0
    while True:
        try:
            out = status(a, 3)
            fails = 0
        except SystemExit as e:   # network blip: keep waiting; give up only after ~30 min of failures
            fails += 1
            print(time.strftime("%H:%M"), f"status failed ({fails}): {str(e)[:120]}", flush=True)
            if fails >= 6:
                raise
            time.sleep(a.every * 60)
            continue
        first = out.splitlines()[0] if out else ""
        print(time.strftime("%H:%M"), first, flush=True)
        if "state running" not in first:
            break
        time.sleep(a.every * 60)
    print(out)
    runs = first.split("runs ")[1].split(" current")[0] if "runs " in first else "[]"
    a.runs = json.loads(runs.replace("'", '"'))
    cmd_fetch(a)
    if a.stop:
        cmd_down(a)


def cmd_down(a):
    # Drive uploads big files in the background: stopping right after a save loses them (the V0_2cam_unfrozen
    # step_015000 checkpoint never reached Drive). Flush first; refuse to stop if that fails, unless --force.
    out = remote("""
import os
if os.path.isdir("/content/drive/MyDrive"):
    from google.colab import drive
    drive.flush_and_unmount()
    print("drive flushed")
else:
    print("drive not mounted")
""", a.session, timeout=1800) if session_exists(a.session) else "no session"
    print(out.strip())
    if "drive flushed" not in out and "not mounted" not in out and not getattr(a, "force", False):
        sys.exit("Drive flush failed; not stopping (use `down --force` to stop anyway)")
    colab("stop", "-s", a.session, capture=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-s", "--session", default="dcad")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("up"); s.add_argument("--gpu", default="A100"); s.set_defaults(f=cmd_up)
    s = sub.add_parser("push-data"); s.add_argument("path"); s.add_argument("--name"); s.add_argument("--gpu", default="A100")
    s.add_argument("--no_sync", action="store_true", help="don't touch the VM's code (e.g. while a job runs)")
    s.set_defaults(f=cmd_push_data)
    s = sub.add_parser("run"); s.add_argument("runs", nargs="+"); s.add_argument("--gpu", default="A100")
    s.add_argument("--stages", default="data,cache,train,probe,eval"); s.set_defaults(f=cmd_run)
    s = sub.add_parser("status"); s.add_argument("--job"); s.add_argument("--lines", type=int, default=12)
    s.set_defaults(f=cmd_status)
    s = sub.add_parser("fetch"); s.add_argument("runs", nargs="+"); s.set_defaults(f=cmd_fetch)
    s = sub.add_parser("wait"); s.add_argument("--job"); s.add_argument("--every", type=float, default=5)
    s.add_argument("--stop", action="store_true"); s.set_defaults(f=cmd_wait)
    s = sub.add_parser("down"); s.add_argument("--force", action="store_true"); s.set_defaults(f=cmd_down)
    a = p.parse_args()
    sys.exit(a.f(a) or 0)


if __name__ == "__main__":
    main()
