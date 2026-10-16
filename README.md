# THIS IS HUMAN WRITTEN! Using Phone Demos to post-train smol vla

I wanted to mess around with post-training, specifically I can imagine a scenario where you may ask a customer to collect some data doing an action so that you can post-train a model to do that task for them. This data is probably not going to be perfect quality/ equivalent to what can be collected in the lab, but you still want to make the most of it, take the phone recorded demos translated into simulation trajectories as the equivalent. I will try to be as sample efficient as possible in post-training SmolVLA (`lerobot/smolvla_base`, 450M).

## The task: pick n place

I played around with having a much less controlled setup, but the amount of data needed(I ran some experiments with perfect sim generated data to see how much was needed) neared 600 episodes so I went with a much more controlled setup so that I could focus on more interesting stuff(one cube, one target zone, a fixed top-down phone, see [Limitations](#limitations)).

TLDR:
**SmolVLA post-trained only on 100 retargeted phone demos has a 95% eval success in simulation so i challenged myself to achieve this with better sample efficiency** (20 episodes each). No robot or sim-expert demos are used.

![phone clip and its sim replay](docs/media/phone_vs_sim.gif)
*Left: one of the 100 phone clips. Right: the sim demo produced from it, rendered from the phone's viewpoint and
aligned in time.*

![policy trained on 100 phone demos](docs/media/rollout.gif)
*The policy trained on all 100 phone demos on an unseen eval layout*

### Success vs number of demos

![success vs number of demos](docs/figures/single_task_demos.png)

Intersetingly I compared the sample efficiency of training on phone demos to training on expert simulation demos that can be deterministically generated. The expert sim demos are much more sample efficient, and the task is saturated with ~15 training demos whereas it takes near 100 phone demos to reach the same accuracy. Thus, the goal from here on became to try and get the accuracy training on 15 phone-demos to be as comparable as possible to training on the expert sim demos

| demos | data | midpoint checkpoint % | end of training % (95% CI) |
|---|---|---|---|
| 15 | phone | 85 | 55 (34–74) |
| 30 | phone | 45 | 82.5 (68–91), 40 episodes |
| 60 | phone | 95 | 80 (58–92) |
| 100 | phone | 100 | 95 (76–99) |
| 15 | clean sim expert | 30 | 95 (76–99) |
| 30 | clean sim expert | 100 | 100 (84–100) |

- Task: "put the blue cube in the red zone". The cube starts anywhere in a 7.7 × 7.7 cm box; the zone is fixed.
  The sim layout distribution matches the real recording layout.
- Phone runs use the first *N* of 100 clips in recording order (nested sets), including the 8% whose sim replay
  failed. Training: from `smolvla_base`, language layers unfrozen, cosine LR over 10k steps (*N* ≤ 30) or 15k steps (*N* > 30) to avoid overfitting.
  Midpoint checkpoint = 5k or 7.5k steps.

## Method comparison at 15 phone demos

At 15 demos naive post-training ends mid-range (55%). I compared several methods to try and boost performance training on the same 15 clips:

- **naive**: flow-matching loss on every clip.
- **ambient diffusion**: the 5 best clips by quality score (top 30% of the 15) are clean; the other 10 supervise only the high-noise part of the flow (per-clip t_min 0.5–0.9 by score rank). The reasoning is that at the beggining of the diffusion denoising process you only need high level reasoning, so a more noisy clip that has the right general idea but say is too jerky/fast/something else can still be used for supervising this part, while towards the end of diffusion you want clean fine-grained demos from supervision of the finegrained/detailed parts of the task.
- **quality-tagged**: at train time assign each clip a quality score that is appended to the state; at test time the policy is queried asking for high quality. Idea is that somewhere in latent space the model begins to associate certain characterstics with bad vs. good performance and then this can then be used at inference. cool idea, credit to pi for this one.
- **velocity-normalized (VN)**: Essentially a hunch that a massive gap between phone demos and sim expert demos is speed(my demos are varying speed while sim is always a fixed speed). I learned about this technique from tutor intelligence's Data Factory 1(DF1) technical report, but its essentially normalizing the speed of all the demos. Naive on clips re-timed to a common speed profile (after Tutor Intelligence DF1,
  Sec. 4.2). Cross-episode cruise-speed CV drops from 0.19 to 0.14; sim replay success rises from 92% to 94%.
- **self-improvement**: A little experiment for funsies. one round on the best of naive / ambient / quality. That run's policy collects 100 sim
  rollouts, the successful ones are added as clean data, and it trains 3k more steps. In hindsight only indistribution would succeed so it wouldnt add much to the model, but was worth a try :).

What the methods see in the data:

| quality tags: highest vs lowest of the 15 clips | ambient: a clean clip vs a noisy one | velocity normalization: as recorded vs normalized |
|---|---|---|
| ![quality tags](docs/media/quality_tags.gif) | ![ambient clean vs noisy](docs/media/ambient_clean_vs_noisy.gif) | ![velocity normalization](docs/media/vn_before_after.gif) |

![final checkpoint by method at 15 phone demos](docs/figures/methods_15.png)
*Final checkpoint of each method on the held-out paired evaluation (fresh layouts; 200 episodes for naive,
ambient and quality-tagged, 100 for velocity-normalized and self-improvement).*

was ultimately able to get accuracy up to 86.5%, which is decent. I reckon maybe by combining several of these(ex. velocity normalization w/ ambient diffusion for example) could maybe climb up a bit more. Also a method such as quality tagging is probably much more influential at mid/pre-training as its probably impossible for the model to create a good latent represenetation of good vs bad demos w/ only 15 labeled clips and 10k steps. For diffusion it capitalizes on having a lot of bad and good data, so could probably extract more out of it at larger data scales.

<!-- METHODS TABLE: python scripts/make_figures.py --methods -->
| method | 2.5k | 5k | 7.5k | 10k (40 ep., 95% CI) | validation, 10k |
|---|---|---|---|---|---|
| *baseline: naive, 5k checkpoint* | | 85 | | | |
| naive | 95 | 85 | 50 | 55.0 (40–69) | 67.5 |
| ambient | 75 | 90 | 85 | **92.5 (80–97)** | 85.0 |
| quality-tagged | 85 | 90 | 80 | 87.5 (74–95) | 80.0 |
| velocity-normalized (naive) | 60 | 85 | 80 | 77.5 (62–88) | 70.0 |
| self-improvement (1 round on ambient) | | | | 67.5 (52–80) at 13k | |
<!-- /METHODS TABLE -->

interstingly training naively sees a large decline after 2.5k steps, likely due to overfitting, and it seems more like all the approaches help to avoid overfitting rather than actually boosting value of the data :(. But a win's a win.

## Future Work
maybe broaden scope of the task so that data quality can be evaluated more directly rather than these methods just helping to avoid overfitting? Also, I wanted to try doing some demo-conditioned work as you may guess from the repo name. Something like icl where the model can be prompted w/ a video and condition the output on that. Unfortunately I am time constrained, compute constrained, and data constrained when it comes to doing that :(. Will probably be adding on to this repo in the future though, I think its an interesting premise to try and extract as much value as possible out of a fixed amount of data in the post-training process. Also, RL-finetuning was an option here but I opted away from it specifically because I wanted to mirror the "quick finetune on customer collected noisy umi-data" as closely as possible with what I had(in this case you probably can't whip up a physically realistic simulation for each customer data collection for rl finetuning, although it could be an intersting premise similar to how people are automating rl env production for llm finetuning?). Anyways, thanks for reading if you made it this far. Below is reproduction stuff that I made claude add.

## Clip Scoring/Quality

Both ambient and quality tagging use one hand-designed score per clip (`scripts/score_phone.py`). It is computed
from the clip's sim replay and compared with clean sim-expert demos of the same task.

**1. Six measurements per clip:**

| measurement | what it is | reference |
|---|---|---|
| grasp error | xy distance between gripper and cube when the gripper first closes | fixed 1.0 cm |
| dwell error | how far the pause before closing differs from the expert's | expert median, 0.2 s allowed |
| lift time | from closing to the cube rising 1 cm | expert 90th percentile |
| jerk | roughness of the path | expert 90th percentile |
| speed | 95th-percentile speed | expert 90th percentile |
| other cubes moved | displacement of non-target cubes | fixed 1.0 cm (always 0 in the single-object scene) |

Dwell, jerk and speed come from the hand track before retargeting inserts its fixed 0.5 s pauses. Each measurement x
becomes a sub-score exp(−max(0, x/ref − 1)): 1 at or better than the reference, decaying when worse.

**2. Score** = 0.4 × (replay succeeded) + 0.6 × (mean of the six sub-scores), in [0, 1].

**3. Quality tag** = 2 × score − 1, in [−1, 1]. It is appended to the state during training; at test time the
policy is given +1.

**4. Clean vs noisy (ambient)** is decided within the clips actually trained on (`--first 15`). The top 30% by score
among successful replays are clean (5 of 15, t_min 0). The rest are noisy, with t_min spread by score rank from
0.5 (best noisy clip) to 0.9 (worst).

**What separates the clips at 15** (`quality/phone_s5_first15.csv`):

- The one failed replay (clip 005) scores 0.31 (tag −0.39); the success term alone puts it last.
- The 14 successful replays score 0.69–0.79, and grasp error does nearly all of the ranking: 0.5–1.3 cm for the
  5 clean clips, 1.3–2.5 cm for the noisy ones. Dwell error (0.3–0.4 s) and lift time barely vary.
- So "noisy" here mostly means "grasped further off-centre", plus the failed replay.

**Caveats:** the weights (0.4 / 0.6), the 30% clean fraction, the 0.5–0.9 t_min range and the references were set
by hand, and the score has not been checked against any independent measure of demo quality. At 15 clips the score
gaps are small: clip 012 (clean, 1.33 cm grasp error) and clip 002 (noisy, 1.31 cm) fall on opposite sides of the
cut.

## Phone pipeline

1. **Recording**: phone 43 cm above the table looking down (0.5× lens); four ArUco markers give
   metric table coordinates in every frame. Session 5: 100 clips of the single-object task, a blue lid moved
   onto a red patch (the sim's blue cube and red zone); the red patch is fixed and the lid starts in a 7.7 × 7.7 cm
   box.
2. **Extraction** (`scripts/process_phone.py`): MediaPipe hand tracking gives the pinch point and gripper open/close.
   Objects are found by color (or one click per clip), with a marker homography per frame.
3. **Retargeting** (`human/retarget.py`): pinch point → gripper target at 10 Hz, contact-anchored height, approach
   from above, dwell at gripper switches. For the thin real lid vs the 4 cm sim cube: a carry lift and a keep-clear
   radius on the retreat. Optional velocity normalization (`human/velocity_norm.py`).
4. **Sim replay**: the retargeted trajectory runs in MuJoCo and is re-rendered from the policy's cameras. Session 5:
   100% of clips extracted, 92% of replays succeed; failed replays are kept.

| phone frame | the sim scene it produced (same instant, phone viewpoint) |
|---|---|
| ![phone frame](docs/media/phone_frame.jpg) | ![sim frame](docs/media/sim_frame.jpg) |

Phone-derived sim demos vs clean sim-expert demos (nearest cube start among the expert demos), front camera:

![phone-derived vs clean sim demo, a](docs/media/phone_replay_vs_clean_a.gif)
![phone-derived vs clean sim demo, b](docs/media/phone_replay_vs_clean_b.gif)

## Limitations

- **Controlled phone setup.** Fixed marker-calibrated top-down camera, flat uncluttered table, known object sizes, a
  recording protocol built for the pipeline, and one object in a small start region. This understates the gap to
  in-the-wild human video.
- **Sim only.** Phone demos are retargeted and re-rendered in sim, so the VLA never sees real pixels and is never run
  on a real robot. This tests action noise from human demos, not the visual domain gap.
- **Easy task.** 15 clean sim demos already give 95%. The single-object result says nothing about grounding or
  multi-object scenes, where SmolVLA failed even with clean data.
- **Statistics.** One training seed per cell, 20 evaluation episodes per point (40 at the final checkpoints of
  the method comparison); 95% intervals of ±15–20 points.

## How to run

Local (Mac, Python 3.11/3.12): `pip install -r requirements.txt`; tests: `python -m pytest -q tests`.
Training runs on Colab (A100) driven from the Mac with the `colab` CLI.

```bash
# sim expert data for the single-object task
python scripts/gen_sim_data.py --scene single --out data/sim_v3s2 --n_per_task 200 --tasks blue-red

# phone session -> retargeted sim replays -> LeRobot dataset -> Drive
python scripts/calib_colors.py data/raw_phone/5 --check_all
python scripts/click_layout.py data/raw_phone/5
python scripts/process_phone.py data/raw_phone/5 --carry_lift 0.015 --retreat_keep_radius 0.10   # -> data/human/5
python vla/export_lerobot.py --src data/human/5 --sources human --tasks all --per_task 1000 --name phone_s5
python colab/dcad.py push-data data/lerobot/phone_s5
python scripts/score_phone.py --datasets phone_s5 --task blue-red --sim_dir data/sim_v3s2 --first 15 \
    --out quality/phone_s5_first15.json                                  # quality scores, clean split
python scripts/vn_stats.py stats --src data/human/5 --out quality/vn_s5.json   # optional: velocity normalization

# SmolVLA on Colab: runs are named in colab/runs.json
python colab/dcad.py up                   # VM + code at the local HEAD; then mount Drive: colab drivemount -s dcad
python colab/dcad.py run P5_naive_15 P5_amb_15 --stages data,cache,train,eval   # the job stops its own VM
python colab/dcad.py wait                 # follow; results: python colab/dcad.py fetch <runs>
python scripts/make_figures.py --single --methods   # README figures
python scripts/make_media.py              # README videos and stills -> docs/media

```
