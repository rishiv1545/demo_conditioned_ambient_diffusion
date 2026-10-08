# Demo-conditioned ambient diffusion (work in progress)

A diffusion policy for a simulated Franka Panda trained on clean sim demos plus phone-recorded human demos. Phase 1: sim environment, phone→robot pipeline and baseline policies.

Full write-up coming. See `NOTES.md` for the running log and `RECORDING.md` for the phone recording protocol.

## Setup (local Mac, the main path)

Developed and run on an Apple Silicon MacBook (M4 Pro). Python 3.11 or 3.12.

```bash
# conda (what we use)
conda create -n dcad python=3.11 -y && conda activate dcad
# ...or a plain venv
python3.11 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

Training picks `cuda`, then `mps`, then `cpu` automatically (override with `--device`). On the M4 Pro, MPS is about 15× faster than CPU for this model. MuJoCo uses its default offscreen renderer on macOS (`MUJOCO_GL=egl` is set only on Linux).

## Run each stage

```bash
python scripts/make_markers.py                                   # Milestone 0: assets/markers.pdf
python -m pytest -q tests                                        # smoke tests
python scripts/eval_expert.py --n 100 --out outputs/m1           # expert success table + videos + speed
python scripts/gen_sim_data.py --out data/sim --n_per_task 50 --tasks all
python policy/train.py --run_name A --data data/sim --sources sim --sim_tasks seen
python policy/evaluate.py --ckpt checkpoints/A/ckpt.pt --k 50 --out outputs/m3/A
```

Multiprocessing scripts take `--workers` (default: core count minus 2).

## Reproduce in Colab

`colab/run_all.ipynb` is a one-click reproduction of baseline A for reviewers: it clones the repo, installs requirements, sets `MUJOCO_GL=egl`, and trains on the Colab GPU (cuda).
