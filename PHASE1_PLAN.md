# Phase 1 Implementation Plan: Environment, Phone Pipeline, Baseline Policy

You are implementing Phase 1 of a robot-learning project for an internship application. Read this whole file before writing code. Work through the milestones in order, and check each milestone's acceptance criteria before moving on.

## Context

**The challenge:** use manipulation data that the applicant personally recorded with a phone to drive a robot arm in simulation. It is judged on creativity, policy performance in sim, and implementation simplicity with a clear presentation. The submission is a public GitHub repo with a README.

**The project (all phases):** a diffusion policy for a Panda arm that is
1. *demo-conditioned*: given one demonstration as context, it performs that task (elementary in-context imitation), and
2. trained with an *ambient diffusion* loss, where phone-derived demos count as corrupted samples and supervise only high-noise diffusion timesteps, while clean sim demos supervise all timesteps.

**Phase 1 (this file)** builds the foundation: the sim environment, the scripted expert, the phone-to-robot data pipeline, and a plain task-conditioned diffusion policy with baseline results. Phase 2 (demo conditioning and the ambient loss) comes later, but Phase 1 code must leave the hooks described under "Hooks for Phase 2."

**Workflow:** all development, training and evaluation run locally on a MacBook with an M4 Pro (Apple Silicon, 12–14 CPU cores, GPU through PyTorch MPS), using a Python 3.11 or 3.12 environment (venv or conda). Code is pushed to GitHub. Colab is only a one-click reproduction notebook for reviewers. Everything must also run on CPU, just slower.

**Platform rules:**
- Device selection: use `cuda` if available, else `mps`, else `cpu`, with a `--device` flag to override. Set `PYTORCH_ENABLE_MPS_FALLBACK=1`. Time one training run on mps and one on cpu, make the faster one the default, and record the timings in `NOTES.md`.
- Rendering: set `MUJOCO_GL=egl` only on Linux. On macOS use MuJoCo's default offscreen renderer.
- Multiprocessing: macOS uses spawn, so every script that uses multiprocessing needs an `if __name__ == "__main__":` guard, and worker functions must be at module level. Multiprocessing scripts take `--workers`, defaulting to the core count minus 2.

## Ground rules

- **Keep it simple.** Plain Python scripts with `argparse`, and plain dataclasses for config. No Hydra, no Lightning, no wandb (CSV logs and matplotlib plots are enough).
- **Use few dependencies:** `mujoco`, `numpy`, `torch`, `opencv-contrib-python` (for ArUco), `mediapipe`, `imageio[ffmpeg]`, `matplotlib`, `tqdm`, `scipy` if needed. Check that mediapipe, mujoco and torch all install on macOS arm64, then pin versions in `requirements.txt`.
- **Never fabricate results.** Every number in the README must come from a script in the repo. If something fails, record it in `NOTES.md`.
- **Keep a running `NOTES.md`** of design decisions, things that failed and why, and the numbers each experiment produced. This becomes the README's "what worked / what didn't" section.
- **Every pipeline stage writes debug visuals** (videos or plots) to `outputs/`, so a human can check it at a glance.
- **Ask the human only when you need physical-world input** (recordings, measurements). Make every other decision yourself and write it down in `NOTES.md`.

## Repo layout

```
.
├── README.md                  # written at the end; keep a stub until then
├── NOTES.md                   # running log of decisions and results
├── RECORDING.md               # recording protocol for the human (Milestone 0)
├── requirements.txt
├── assets/
│   ├── scene.xml              # tabletop scene including the menagerie Panda
│   └── markers.pdf            # printable ArUco markers
├── sim/
│   ├── env.py                 # PickPlaceEnv
│   ├── ik.py                  # damped least-squares IK
│   └── expert.py              # scripted expert
├── human/
│   ├── calibrate.py           # ArUco homography and height calibration
│   ├── extract.py             # MediaPipe hand tracking, object/zone detection
│   ├── retarget.py            # human trajectory -> robot EE trajectory
│   └── replay.py              # replay in sim -> (obs, action) dataset
├── policy/
│   ├── data.py                # dataset loading, normalization, chunking
│   ├── model.py               # noise-prediction network
│   ├── diffusion.py           # DDPM training loss, DDIM sampling
│   ├── train.py
│   └── evaluate.py
├── scripts/                   # thin entry points, e.g. gen_sim_data.py, make_figures.py
├── tests/                     # quick smoke tests
├── colab/run_all.ipynb        # one-click reproduction for reviewers; thin
└── data/, outputs/, checkpoints/   # gitignored (except small example outputs for the README)
```

## The task family

Design this carefully, because Phase 2's in-context experiments depend on it.

- A table with **3 colored cubes** (red, green, blue, about 4 cm) and **3 target zones** (flat colored squares, about 10 cm; use colors different from the cubes, e.g. yellow, purple, orange).
- **Task = (cube c, zone z)**: pick cube c and place it in zone z. That makes 9 tasks.
- Cube and zone positions are randomized each episode inside a workspace rectangle (start with 50 × 35 cm), with minimum spacing so nothing overlaps.
- **Success:** at the end of the episode, the target cube's center lies inside the target zone, the cube rests on the table, and the gripper has released it.
- **Held-out split:** clean sim demos exist for only 6 of the 9 tasks. Hold out 3 tasks chosen so that each cube and each zone still appears in some seen task (a Latin-square pattern, e.g. (red, yellow), (green, purple), (blue, orange)). Phone demos cover all 9 tasks, with extra demos for the held-out ones. This is what makes the phone data matter: it is the only source for the held-out tasks.
- Make the split configurable, and make the number of sim demos per task configurable (low-data regimes such as 5, 10 or 20 per task matter for later ablations).

## Milestone 0: Recording protocol (do this first)

The human needs to record phone demos while you build the sim. Before any other code, produce:

1. **`assets/markers.pdf`**: four ArUco markers (`DICT_4X4_50`, IDs 0–3), each with a printed side length of 6 cm and the ID and size labeled, one per A4 page or two per page. Generate it with OpenCV and matplotlib.
2. **`RECORDING.md`**: a short, concrete checklist for the human:
   - **Physical setup:** tape the 4 markers at the corners of a measured 50 × 35 cm rectangle on the table (marker centers at the corners; record the actual measured distances). Use any 3 distinctly colored, pinchable objects (2–6 cm) and any 3 flat colored patches (8–10 cm); household items are fine (Lego, dice, erasers, sticky notes, printed colored squares). `session.json` maps the canonical task names (red/green/blue, yellow/purple/orange) to the real items, with their heights and HSV ranges.
   - **Camera:** the phone looks roughly straight down from 60–90 cm (no-purchase mounts: taped under a shelf, a broom handle across two chairs, or, as a last resort, a tall book stack at an angle). Mount it rigidly, use landscape orientation, 1080p at 30 fps, and lock focus and exposure. Write down the camera height. All 4 markers must be visible throughout.
   - **A calibration clip at the start of each session:** about 3 s with the hand resting flat on the table at a "home" spot (the bottom-right corner, say), then about 3 s with the hand held still on top of a box of known, measured height. This calibrates the height estimate.
   - **Each demo clip:** start with the hand at home, pick up the specified cube using a thumb–index pinch from above, place it in the specified zone, release, and return home. Keep each clip to about 5–15 s. Randomize cube and zone positions between clips.
   - **How many:** 4 demos per seen task (24) plus 8 per held-out task (24), 48 in total. The file name encodes the task, e.g. `red-yellow_03.mp4`.
   - **Tips:** good even lighting, no other red, green or blue clutter in view, and a long sleeve is fine but keep the hand itself uncovered.
   - Where to put the files: `data/raw_phone/<session_name>/` on the Mac, along with `session.json` containing the camera height, the rectangle measurements and the calibration box height.

Tell the human Milestone 0 is ready (one short message) before continuing.

## Milestone 1: Simulation environment and scripted expert

**Sim choice: custom MuJoCo scene with the Franka Panda from `mujoco_menagerie`.** Get the model files by cloning the menagerie or vendoring only the `franka_emika_panda` folder (check its license and keep it). Do not use LIBERO or robosuite. This task family needs full control of objects, zones and randomization, and fast headless rollouts on a laptop CPU. Record this rationale in `NOTES.md`.

**`sim/env.py` (`PickPlaceEnv`):**
- `reset(task=(c, z), seed=..., layout=None)`: randomizes the layout, or uses a given layout (needed to replay human demos with the objects where they were in the real video).
- **Control at 10 Hz.** Physics runs at the MuJoCo default timestep, with substeps between control steps.
- **Action = absolute end-effector target position `(x, y, z)` in the table frame, plus a gripper command in [0, 1]** (0 = open, 1 = closed). The gripper orientation is fixed pointing straight down with fixed yaw. Cubes spawn axis-aligned (no yaw randomization in v1).
- Use absolute EE targets, not deltas. This makes retargeted human trajectories much easier to use.
- **`sim/ik.py`:** damped least-squares IK on the hand site's position and orientation, converting the EE target into joint position targets for the menagerie position actuators. Clip the per-step EE motion so a far-away target doesn't produce violent motion. Hand-roll this in about 40 lines (no extra IK library).
- **Observation (a dict, also flattened to a vector):** EE position (3), gripper opening (1), the position of each cube (3 × 3) and of each zone (3 × 2, xy only). Use a fixed object order (red, green, blue / yellow, purple, orange) so the observation does not reveal the task.
- **Task conditioning in Phase 1:** a 6-dimensional vector (cube one-hot plus zone one-hot), kept separate from the observation.
- `render(camera="top"|"front")` for videos. Set `MUJOCO_GL=egl` only on Linux (Colab); on macOS use MuJoCo's default offscreen renderer.
- Tune friction and gripper force until grasps are reliable, and note the values in `NOTES.md`.

**`sim/expert.py`:** a waypoint state machine: move above the cube, descend, close, lift, move above the zone, descend, open, lift, then hold. Add diversity so the dataset is not one fixed trajectory: random hover heights, waypoint jitter of a few mm, random speed scaling and small random pauses.

**`scripts/gen_sim_data.py`:** generates N episodes per task for the configured task split and saves them in the episode format below.

**Acceptance criteria:**
- The expert succeeds on at least 95% of 100 random episodes for every one of the 9 tasks. Print the table and save it.
- `outputs/m1/expert_<task>.mp4` shows clean pick-and-place for a few tasks.
- A headless rollout of 200 steps is fast enough for evaluation (measure and note steps per second).
- `tests/test_env.py`: reset, step, success detection, and that the expert succeeds on a fixed seed.

## Episode format (shared by sim and human data)

One `.npz` file per episode, or one file per dataset of concatenated episodes with an index. Each episode stores:

- `obs` with shape [T, obs_dim]; `action` with shape [T, 4]; `task` as (cube index, zone index)
- `source`: `"sim"` or `"human"`
- `layout`: the initial object and zone positions
- `success`: whether the episode achieved the task (for human data, whether the *sim replay* succeeded)
- `sigma_n`: a float. 0.0 for sim; for human data, NaN until Phase 2 estimates it
- `meta`: a dict containing the human clip file name, the replay tracking error, etc.
- For human episodes also store `raw_traj` (the retargeted EE trajectory before replay) and `raw_gripper`.

Write a single loader `policy/data.py:load_episodes(paths, filters)` that every later stage uses.

## Milestone 2: Phone-to-robot pipeline

Build it in this order, with a debug visual at every step. Develop against the first real clips as soon as the human provides them. Until then, test on any hand video, or on a short clip you ask the human to record.

1. **`human/calibrate.py`:** detect the 4 ArUco markers in each frame and compute the image-to-table homography (frame by frame, since the phone may shift slightly; smooth it over time). Use the measured rectangle from `session.json`. Debug output: the warped top-down view with the table grid drawn on it.

2. **`human/extract.py`:**
   - Hand tracking with MediaPipe. Use the current **Tasks API `HandLandmarker`** (the legacy `mp.solutions.hands` API is deprecated; check the current documentation and model download URL).
   - **Grasp point** = the midpoint of the thumb tip and index fingertip in the image, mapped to the table plane with the homography. That gives xy.
   - **Height z from apparent hand size.** For a camera at height H looking down, apparent size s scales as 1/(H − z). Measure the pixel length of a rigid hand segment (wrist to index MCP, say, or the palm width), take s₀ from the "hand flat on table" calibration, and compute z = H·(1 − s₀/s). Check it against the known-box-height calibration and report the error in `NOTES.md` whatever it is. The camera may be at an angle: the homography handles xy, but this height model assumes a roughly top-down view, which `NOTES.md` must state. Also apply a parallax correction to xy using z, since the grasp point sits above the table plane.
   - **Gripper signal:** the thumb–index distance normalized by hand size, thresholded with hysteresis, then cleaned with a minimum-duration filter.
   - **Object and zone detection:** HSV color thresholding on the warped first frame gives the initial object and zone positions, using per-session HSV ranges from `session.json`. A click tool (`scripts/calib_colors.py`) fits these ranges from one click per item. **Manual fallback:** `scripts/click_layout.py` lets the human click the 6 positions on each clip's warped first frame; the resulting per-clip JSON overrides color detection. Optionally detect them again on the last frame to label whether the human actually completed the task.
   - Smooth trajectories (Savitzky–Golay or a one-euro filter) and fill short tracking dropouts by interpolation; mark long dropouts.
   - Debug output: the phone video with landmarks, the grasp point, the estimated z and the gripper state overlaid, plus a plot of x, y, z and gripper over time.

3. **`human/retarget.py`:** map the table-frame grasp-point trajectory to an EE target trajectory in the sim table frame. Use 1:1 metric scale with a fixed offset (the real rectangle is defined to match the sim workspace), add a fixed z offset between the human pinch point and the Panda fingertip center, resample to 10 Hz, and convert the gripper signal to the action's [0, 1] range. Clip to the workspace.

4. **`human/replay.py`:** reset the sim with the detected layout and execute the retargeted trajectory open-loop as actions. Record the resulting (obs, action) pairs in the episode format. Store `success` and the tracking error (commanded vs achieved EE position). Some replays will fail (a missed grasp, say). **Keep the failures in the dataset, flagged.** They are exactly the noisy data Phase 2's ambient loss is designed for.

5. **`scripts/process_phone.py`:** runs steps 1–4 over a session folder and prints a summary table: per task, the number of clips, the extraction success rate, the replay success rate and the mean tracking error.

**Acceptance criteria:**
- A side-by-side video, `outputs/m2/<clip>_side_by_side.mp4`, showing the phone clip with overlays on the left and the sim replay on the right, for at least 3 clips.
- The summary table exists, along with the height-estimation error from calibration.
- Replay success across all clips is reported, whatever it is. If it is below about 50%, investigate the biggest cause (probably the z estimate or grasp timing), make at most one or two fixes, and record before and after numbers. Don't chase perfection: noisy human data is expected and is part of the story.

## Milestone 3: Baseline diffusion policy

**Model (`policy/model.py`, `policy/diffusion.py`):**
- Predicts an action chunk of horizon H = 16 at 10 Hz; at execution, run the first 8 actions, then re-plan.
- Conditioning: the current observation (optionally the last 2 frames) plus the 6-dimensional task vector, embedded by an MLP and injected with FiLM.
- Network: a small 1D temporal U-Net (Chi et al.'s Diffusion Policy style), or a small transformer over the action tokens. Pick one and keep it at about 1–5M parameters.
- **Hand-roll DDPM** (about 100 training steps, cosine schedule, ε-prediction) **with DDIM sampling** (10 steps). Do not use `diffusers`: Phase 2 has to modify the loss per sample, and a hand-rolled implementation is short and transparent.
- Normalize observations and actions to [-1, 1] using statistics from the training set; save the statistics with the checkpoint.

**`policy/train.py`:**
- `--data` takes a list of dataset paths plus filters: sources, which tasks, whether to include failed human replays.
- AdamW, an EMA of the weights, and a fixed number of steps. Log loss to CSV and save checkpoints to `checkpoints/<run_name>/`.
- It must finish in under about 30 minutes on the M4 Pro for the default dataset size.

**`policy/evaluate.py`:**
- For each of the 9 tasks, run K episodes (default 50) on fixed held-out seeds, headless and parallelized with `multiprocessing` across CPU cores (`--workers`, default core count minus 2).
- Output: a per-task success CSV, the mean success on seen vs held-out tasks with 95% confidence intervals, and rendered videos of a few episodes (both successes and failures).

**Baseline runs (the deliverable for Phase 1):**
- **A — sim only:** clean sim demos for the 6 seen tasks.
- **B — sim + human, naive:** A's data plus all human replay episodes, treated exactly like sim data.
- **B′ — B with only successful human replays.**
- Report all three in one table: seen-task success and held-out-task success. Generate a bar chart with `scripts/make_figures.py`.

The expected outcome: A does well on seen tasks and poorly on held-out ones, while B and B′ give some held-out success but may hurt seen-task performance because of noisy data. Whatever happens, report it honestly. This gap is what Phase 2 attacks.

**Acceptance criteria:** the three runs are trained and evaluated locally, and the table, the plot and example videos exist in `outputs/m3/`. The Colab notebook reproduces A end to end from a fresh runtime.

## Hooks for Phase 2

Phase 1 must not implement these, but the code must make them easy to add:

1. **A per-sample minimum timestep in the loss.** `diffusion.loss(model, x0, cond, t_min=None)` samples t uniformly from [t_min_i, T] for each sample i. With `t_min=None` it behaves like standard DDPM. Phase 2 sets t_min from each episode's `sigma_n` (zero for sim data, so sim samples cover every timestep).
2. **Conditioning through one interface.** The model takes a `cond` tensor produced by a swappable `ConditionEncoder`. In Phase 1 this encodes the observation and the task one-hot. Phase 2 will add an encoder that takes the observation plus a context demonstration (a subsampled trajectory of about 16 keyframes) instead of the one-hot. Keep the observation encoding and the task encoding separate inside it.
3. **Context sampling.** `data.py` should be able to return, for any training sample, another episode of the same task (a different episode, from a configurable source). Implement the sampler and test it; don't use it in training yet.
4. **`sigma_n` flows through** from the episode files to each batch.

## Colab notebook (`colab/run_all.ipynb`)

A one-click reproduction for reviewers only; the main path is local. Keep it thin. Each cell calls a script:
1. Clone the repo, `pip install -r requirements.txt`, set `MUJOCO_GL=egl`.
2. Generate sim data, train on cuda, evaluate, make figures (baseline A).

## Docs

The README's main path is the local Mac setup (venv or conda, install, run each stage). Colab gets one short "Reproduce in Colab" section.

## Definition of done for Phase 1

- [ ] Milestone 0 delivered and the human told
- [ ] The expert reaches at least 95% success on all 9 tasks; the env tests pass
- [ ] Phone pipeline: side-by-side videos plus the summary table
- [ ] Baselines A, B and B′ evaluated, with the table and plot saved
- [ ] All four Phase 2 hooks exist and have a smoke test each
- [ ] `NOTES.md` is current; the README stub lists how to run each stage
- [ ] Code pushed; the README documents the local Mac path; the Colab notebook runs baseline A from a fresh runtime

Finish by writing a short summary for the human: the key numbers, what went wrong, and anything that looks risky for Phase 2.
