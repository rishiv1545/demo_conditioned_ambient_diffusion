# A little robot data + cheap phone data: can noisy human demos help post-training?

*Work in progress. Running log: `NOTES.md`. Plan: `PHASE2_PLAN.md`. Phone recording protocol: `RECORDING.md`.*

## The idea

**A little robot data plus cheap human phone data.** Robot demonstrations (teleoperation) are expensive, so a new task usually gets only a handful. Recording a human doing the task with a phone is cheap, but the result is noisy: the hand is tracked imperfectly, a human grasps differently from a parallel gripper, and the retargeted trajectories miss by centimeters. The question is whether such phone demos **improve** post-training when robot data for a new task is scarce, and whether an **ambient diffusion / flow-matching loss** (noisy samples supervise only the high-noise end of the diffusion process) makes them usable where naive training on them does not.

The test bed is a simulated Franka Panda picking one of three colored cubes and placing it in one of three zones (9 tasks). A **scripted sim expert stands in for teleoperation**: it provides the "robot data", plenty for the red and green tasks, and a few demos (or none) for the tasks with the **held-out blue cube**, which is in every scene as a distractor but is never the target of a robot demo beyond those few. The phone data covers all tasks, blue included: 48 clips recorded on a real table and retargeted into the sim.

## Headline: phone-data ablations (small diffusion policy, 3 seeds)

Blue = held-out tasks; seen = red/green. Robot data: 20 sim-expert demos per seen task, plus *K* per blue task.

| robot demos per blue task | phone demos (48 replays) | loss on phone data | seen % | blue % |
|---|---|---|---|---|
| 0 | none | – | 88.7 (1 seed) | **0.0** |
| 0 | all | naive | 86.0 ± 1.2 | 9.6 ± 2.2 |
| 4 | none | – | 87.6 ± 0.6 | 44.4 ± 3.5 |
| 4 | all | naive | 82.8 ± 0.3 | 43.8 ± 3.3 |
| 4 | 37 successful replays only (filtered) | naive | 85.2 ± 0.9 | 50.7 ± 3.8 |
| 4 | 37 successful replays only (filtered) | ambient | 91.0 ± 0.3 | 64.9 ± 0.3 |
| 4 | **all** | **ambient** | **90.0 ± 1.9** | **65.3 ± 1.1** |

- **Without phone data, the blue tasks are at 0%** (no robot demo of the blue cube), and the same holds for SmolVLA (0% blue after sim-only post-training, see below).
- With 4 robot demos per blue task, **phone data with the ambient loss raises blue success from 44% to 65%**, while naive training on the same phone data gains nothing (44%) and costs seen-task performance.
- **The ambient loss beats filtering:** dropping the 11 replays that failed in sim and training naively reaches 51%; the ambient loss reaches 65% **with or without** the failed replays, so the phone demos need no curation.
- Phone data alone (0 robot demos per blue task) gives only 10%: the noisy demos help by refining a little clean data, not by replacing it.

## Setup: phone → retargeting → sim

<!-- TODO: short paragraph per stage + numbers from NOTES.md (calibration, tracking, replay success) -->

1. **Recording** (`RECORDING.md`): a phone looks down at the table (43 cm, 0.5× lens); four ArUco markers give metric table coordinates in every frame; 48 clips (4 per red/green task, 8 per blue task).
2. **Extraction** (`scripts/process_phone.py`): MediaPipe hand tracking → pinch point and gripper open/close; objects from color or one click per clip; marker homographies per frame.
3. **Retargeting** (`human/retarget.py`): pinch point → gripper target at 10 Hz, contact-anchored height, dwell at gripper switches, and an **approach-from-above rule** (a human hand slides in low from the side; the parallel gripper must come down on the object). With the rule, sim replay success went from 44% to **77% (37/48)**.
4. **Replay in sim**: the retargeted trajectory is executed in MuJoCo and re-rendered; successful and failed replays are both kept as noisy demos.

![phone recording vs sim replay](docs/figures/phone_side_by_side.jpg)
*Phone recording with hand tracking (left), the sim replay from the phone's viewpoint (middle) and from the front (right). Videos for all 48 clips: `outputs/m2/3/*_side_by_side.mp4`.*

## Small diffusion policy (mechanism study)

A 5.4M-parameter DDPM policy (1D temporal U-Net, FiLM-conditioned on privileged state: gripper, cube and zone positions, plus a one-hot task) isolates the data/loss question from perception and language. Training data: 20 clean sim demos per red/green task, plus *K* clean demos per blue task, plus noisy demos. Ambient loss: noisy samples draw the diffusion step from [t_min, T). Evaluation: 50 episodes per task, fixed seeds.

### Real phone replays

![real phone replays on the small policy](docs/figures/phone_small_policy.png)

With 4 clean blue demos per task, adding the 48 phone replays **naively** changes nothing on blue (44 → 44%) and costs seen performance; with the **ambient loss (t_min 75)** they raise blue success to **65%** (3 seeds each). Control (hatched bars): keeping only the 37 replays that succeeded in sim gives 51% with the naive loss and 65% with the ambient loss. So the ambient loss beats filtering out the failed replays, and it is insensitive to whether they are included. The gain is spread evenly across the three blue tasks (+17 to +25 points each).

### The best t_min rises with the noise level

![blue success vs t_min for three noise levels](docs/figures/noise_vs_tmin.png)

Synthetic phone-like noise (per-episode grasp offset, height offset, jitter, gripper timing) at three levels. With mild noise the naive loss is best and the ambient loss only discards signal; with phone-level and stronger noise, the high-t_min end is best.

### Clean data vs noisy data

![clean scaling](docs/figures/clean_scaling.png)

With 1× synthetic noise, the noisy demos (with the ambient loss) beat discarding them only at 4 clean demos per task; from 8 up, discarding is better, although the ambient loss recovers most of what naive training loses.

All numbers: [`docs/results_tables.md`](docs/results_tables.md) (generated by `python scripts/make_figures.py --results`).

## SmolVLA post-training

<!-- TODO: fill as the runs complete -->

- **Model and pipeline:** `lerobot/smolvla_base` (450M) post-trained from two 512 px cameras (scene + wrist), robot state and a language instruction; language layers unfrozen (SigLIP and connector frozen, so vision features are cached). Training runs on Colab A100s driven from the Mac (`colab/dcad.py`; run registry `colab/runs.json`; resumable jobs with all state on Drive; several VMs in parallel).
- **Diagnosis of the base policy** (sim only, 30k steps: 39% seen, 0% blue): 34% of seen episodes are missed grasps (vs 4% wrong cube), so seen performance is limited by grasp precision, not grounding; raw-image and cached-feature inference are bit-identical; the wrist camera shows the target clearly at the grasp; the flow-sampling spread (1.6 cm) is comparable to the grasp tolerance. The policy does not ground "blue" zero-shot (probe accuracy 0.03–0.12, below chance).
- **Base-policy fixes** (seen tasks, 120 episodes per cell, 95% CI about ±9 points):

  | checkpoint | sampling | success % | missed grasp % |
  |---|---|---|---|
  | V0 (30k steps) | normal | 39.2 | 34.2 |
  | V0 (30k steps) | zero noise | 43.3 | 33.3 |
  | + 3k steps of LR decay | normal | 40.8 | 34.2 |
  | + 3k steps of LR decay | zero noise | 45.8 | 29.2 |
  | + recovery demos + LR decay | normal | 54.2 | 22.5 |
  | + recovery demos + LR decay | zero noise | **65.0** | **16.7** |

  Recovery demos (DART-style: the scripted expert's path is perturbed by 1–3 cm and labeled with the nominal targets that lead back to it) are what moves closed-loop success; LR decay alone halves the training loss but changes nothing. Grasp diagnostics (`vla/grasp_diag.py`) explain why: at the gripper close, the policy's offset from the cube is **scatter, not bias** (mean 0.7–1.3 cm, no stable direction across checkpoints, vs a 2–2.7 cm spread), and the **offline** grasp-point error on training frames is small (0.26 cm median after decay) while the **closed-loop** error is large (rms 3.65 cm). The errors come from compounding drift in closed loop, which recovery data targets (rms 3.65 → 2.62 cm).
- **Scarce-robot-data runs (running):** from the recovery+decay checkpoint, +4k steps with 4 sim-expert demos per blue task (C), + the 48 phone replays naively (C+N), and with the ambient loss (C+N+amb, t_min 0.72 = small-policy t_min 75, matched by noise-to-signal ratio); seen-task recovery data kept, task-balanced sampling, zero-noise evaluation (50 episodes per blue task). The earlier C runs (`C_blue4`, `CN_blue4`: 2–3% blue) sampled uniformly, so the blue demos were 1% of the samples; they are not comparable.
- **Phone recovery demos (planned):** "miss and correct" phone clips (aim beside the object, pause, correct) with the loss masked before the correction and the ambient loss, replacing the sim recovery data in the recovery+decay recipe (`V0_30k_phonerec_decay3k`), to test whether cheap human corrections can stand in for robot recovery data.

## Limitations

- **Controlled phone setup.** The phone data comes from a fixed, marker-calibrated top-down camera, a flat uncluttered table, known object sizes and a recording protocol designed for the pipeline. Its noise therefore understates the gap to in-the-wild human video (moving cameras, no metric scale, occlusion, clutter).
- Phone demos are retargeted and re-rendered in sim, so the VLA never sees real phone pixels: this tests demonstration (action) noise, not the visual domain gap.
- **Single-seed VLA runs**, with 20 evaluation episodes per task (95% intervals of about ±10–15 points); the small policy has 3 seeds for the key cells only.
- One task family and one held-out object.

## Setup (local Mac)

Developed on an Apple Silicon MacBook (M4 Pro). Python 3.11 or 3.12.

```bash
conda create -n dcad python=3.11 -y && conda activate dcad     # or: python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Training picks `cuda`, then `mps`, then `cpu` (override with `--device`). MuJoCo uses its default offscreen renderer on macOS (`MUJOCO_GL=egl` on Linux).

## Run each stage

```bash
python -m pytest -q tests                                        # tests
python scripts/gen_sim_data.py --out data/sim_v2 --n_per_task 200 --tasks all
python policy/train.py --run_name A --data data/sim_v2 --sources sim --sim_tasks seen
python policy/evaluate.py --ckpt checkpoints/A/ckpt.pt --k 50 --out outputs/m3/A
python scripts/run_ambient_priority.py                           # small-policy ambient sweep (Mac)
python scripts/make_figures.py --results                         # README figures + tables
```

Phone data (see `RECORDING.md`):

```bash
python scripts/calib_colors.py data/raw_phone/<session> --check_all
python scripts/click_layout.py data/raw_phone/<session>
scripts/phone_to_colab.sh data/raw_phone/<session>               # process -> LeRobot dataset -> Drive
```

SmolVLA on Colab (A100), from the Mac:

```bash
python colab/dcad.py up                       # VM + code at the local HEAD; then `colab drivemount -s dcad`
python colab/dcad.py run <runs from colab/runs.json>
python colab/dcad.py wait --stop              # fetch results, flush Drive, stop the VM
```
