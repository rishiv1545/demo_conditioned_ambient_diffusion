# Phone recording protocol

Read this once before you start, then use it as a checklist. A session takes about 45–60 minutes, plus about 5 minutes of clicking afterwards (section 8).

## 1. Things you need

- [ ] `assets/markers.pdf` printed **at 100% / actual size** (no "fit to page"). Use a ruler to check that each black square is 6.0 cm. Cut the markers out with a white margin around each one.
- [ ] **3 objects**, any you can pinch between thumb and index finger, **2–6 cm** in size, each a **different, distinct color**. Household options: Lego/Duplo bricks, dice, erasers, bottle caps, small toy blocks, a matchbox. Squarish objects that sit stably work best.
- [ ] **3 flat colored patches**, **8–10 cm** across, again three distinct colors. Household options: sticky notes (2–3 stacked if small), colored paper or card, coasters, or squares printed in solid color.
- [ ] A box (or book stack) of known height, 10–20 cm, with a flat top.
- [ ] A tape measure and tape.
- [ ] A phone mount; no purchase needed (see section 3).

**Colors:** the colors don't have to be red/green/blue or yellow/purple/orange. Each of your 6 items is mapped to a task name in `session.json` (section 7). File names always use the task names: if your "red" is a green eraser, a clip where you move the eraser to the "yellow" patch is still `red-yellow_NN.mp4`. Pick 6 colors that are clearly different from each other, from the table and from your skin. Saturated colors are easiest. Avoid white, gray, black, skin tones and wood colors if you can; if you can't, the manual click fallback (section 8) handles it.

Write down which object stands for which name before you start:

| task name | your object | | task name | your patch |
|---|---|---|---|---|
| red | _____________ | | yellow | _____________ |
| green | _____________ | | purple | _____________ |
| blue | _____________ | | orange | _____________ |

## 2. Table frame (important: the sim uses the same frame)

Stand at the **near edge** of the table, where you'll stand while recording. Tape the markers so their **centers** sit at the corners of a 50 × 35 cm rectangle:

```
        far edge (away from you)
   ID 3 ●──────── 50 cm ────────● ID 2
        │                       │
      35 cm      workspace     35 cm
        │                       │
   ID 0 ●──────── 50 cm ────────● ID 1
        near edge (you stand here)       [HOME] = just inside ID 1, near-right corner
```

- x runs from ID 0 toward ID 1 (to your right), y runs from ID 0 toward ID 3 (away from you), and the origin is the center of ID 0.
- Keep the markers flat and unwrinkled, all the same way up (the text label toward you).
- **Measure the actual center-to-center distances** 0→1, 1→2, 2→3, 3→0 and the two diagonals 0→2 and 1→3, and write them into `session.json` (below). A few mm off from 50 × 35 is fine, as long as you record it.
- **HOME** is a spot about 5 cm inside the rectangle from ID 1 (near-right). Mark it with a small piece of tape.

## 3. Camera

The phone should look **down at the table from 60–90 cm**. Straight down is best: the height estimate assumes a roughly top-down view. A moderate angle (up to about 30° from vertical) is fine, since the markers correct the perspective, but expect a larger height error. No-purchase mounts:

- **Taped under a shelf** or cabinet above the table, camera facing down. This is the most rigid option.
- **A broom handle across two chairs** (or two stacks of books) above the table, with the phone taped to the middle of it facing down. Weigh the chair seats down so nothing moves.
- **Last resort: a tall book stack** at the far or side edge of the table, with the phone leaning over the edge at an angle. Make sure it's rigid and all 4 markers are in view.

Checklist:
- [ ] Mounted rigidly; the phone must not move during a clip. Small shifts between clips are fine (the markers are tracked in every frame).
- [ ] Landscape orientation, **1080p at 30 fps**.
- [ ] **Lock focus and exposure** (on iPhone, long-press on the table until "AE/AF LOCK" shows; on Android, use Pro mode or the lock option).
- [ ] All 4 markers stay **fully visible in every frame**, with some margin. Check the edges of the picture before you start.
- [ ] Measure the camera height: the **vertical** distance from the lens to the table surface, in cm.

## 4. Calibration clips (once per session, before the demos)

Hand pose: hold your right hand **flat, palm down, fingers together**, at the same angle you'll pinch with, i.e. back of the hand facing the camera.

1. `calib_table.mp4`: about 3 s with your hand resting flat on the table at HOME, not moving.
2. `calib_box.mp4`: put the box inside the rectangle, then about 3 s with your hand resting flat on top of the box, not moving. Measure and record the box height.

## 5. Demo clips

For each clip:

1. **Randomize the layout:** place the 3 objects and 3 patches at random spots inside the rectangle. Keep at least 8 cm between any two items and at least 3 cm from the markers. Keep objects square to the rectangle where possible (not rotated).
2. Start recording. **Hand at HOME**, still, for about 1 s.
3. Using your **right hand** with a **thumb–index pinch from above**, pick up the named object, carry it, place it **in the middle** of the named patch, and release.
4. **Return the hand to HOME**, hold for about 1 s, then stop recording.
5. Keep the hand **open (fingers spread)** whenever you're not holding an object, and pinch decisively, so the gripper signal is clear.

Each clip should be about **5–15 s**. Move at a calm, natural speed. Keep your arm and sleeve out of the way of the objects where you can; the hand itself must stay uncovered. **In the first second of each clip, the hand must not cover any object or patch:** the layout is read from the first frame.

### How many clips, and file names

File name: `<object>-<patch>_<NN>.mp4` using the **task names**, e.g. `red-yellow_03.mp4`.

| Task | Clips | | Task | Clips | | Task | Clips |
|---|---|---|---|---|---|---|---|
| **red-yellow** (held-out) | **8** | | red-purple | 4 | | red-orange | 4 |
| green-yellow | 4 | | **green-purple** (held-out) | **8** | | green-orange | 4 |
| blue-yellow | 4 | | blue-purple | 4 | | **blue-orange** (held-out) | **8** |

**48 clips in total.** If a clip goes wrong (the object is dropped, a marker gets covered), just re-record it under the same name. You don't need to keep the bad one.

## 6. Tips

- Use good, even lighting with no hard shadows; daylight plus room lights works well. Avoid glare on the markers and on shiny objects.
- Keep anything else in your 6 colors out of the camera view, including clothing, mugs and phone cases.
- A long sleeve is fine, but keep the hand itself uncovered (no gloves, and preferably no bright rings or watches).
- Only one hand in view.

## 7. Where to put the files

```
data/raw_phone/<session_name>/        e.g. data/raw_phone/2026-10-10_kitchen/
├── session.json
├── calib_table.mp4
├── calib_box.mp4
├── red-yellow_01.mp4
└── ...
```

`session.json`: fill in your measurements (cm) and what each task name is. `height_cm` is the object's height as it sits on the table. Leave out `"hsv"`; the color tool in section 8 fills it in.

```json
{
  "camera_height_cm": 75.0,
  "rect_cm": {"d01": 50.0, "d12": 35.0, "d23": 50.0, "d30": 35.0, "d02": 61.0, "d13": 61.0},
  "calib_box_height_cm": 15.0,
  "hand": "right",
  "objects": [
    {"name": "red",   "label": "red Lego 2x2 brick", "height_cm": 2.0},
    {"name": "green", "label": "green eraser",       "height_cm": 1.5},
    {"name": "blue",  "label": "blue die",           "height_cm": 1.6}
  ],
  "zones": [
    {"name": "yellow", "label": "yellow sticky note"},
    {"name": "purple", "label": "pink card square"},
    {"name": "orange", "label": "orange coaster"}
  ],
  "notes": "camera taped under shelf, about 10 deg tilt"
}
```

Copy the session folder to `data/raw_phone/` on the Mac that runs the pipeline. It's gitignored, so videos never go into the repo.

## 8. After recording: colors, or clicks (on the Mac)

1. **Color calibration (about 1 minute):**
   `python scripts/calib_colors.py data/raw_phone/<session> --check_all`
   A top-down view of the first clip opens. Click once on each item in the order shown (red, green, blue objects, then yellow, purple, orange patches), pick a well-lit spot, then press Enter. The tool saves HSV ranges into `session.json`, then reports which clips have all 6 items detected.
2. **Manual fallback, if detection misses items (about 5 minutes for 48 clips):**
   `python scripts/click_layout.py data/raw_phone/<session>`
   For each clip it shows the top-down first frame. Click the **center of the top** of each object, then the **center** of each patch, in the same order, and press Enter. `u` undoes, `s` skips, and `q` quits (progress is saved). Each clip gets a `<clip>.layout.json` file, which overrides color detection. Already-clicked clips are skipped on the next run; `--redo` re-clicks them.
3. Run the pipeline: `python scripts/process_phone.py data/raw_phone/<session>`.
