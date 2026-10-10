"""PickPlaceEnv: Panda + 3 colored cubes + 3 colored target zones on a table.

World frame == table frame (see assets/scene.xml). Control runs at 10 Hz; an action is an absolute end-effector
target (x, y, z) for the fingertip-center site "ee" plus a gripper command in [0, 1] (0 = open, 1 = closed).
"""
import os
import sys
from dataclasses import dataclass

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from sim.ik import IK, down_rot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "assets", "scene.xml")

CUBES = ("red", "green", "blue")
ZONES = ("yellow", "purple", "orange")
ALL_TASKS = [(c, z) for c in range(3) for z in range(3)]
# Held-out task splits. "object" (default): the blue cube is never a target in sim demos (it is still in every
# scene as a distractor), so phone demos are the only source for the blue tasks. "combo": a Latin square of
# (cube, zone) pairs where every cube and zone still appears in some seen task. A policy with factored
# conditioning composes these for free (100% held-out from sim alone; see NOTES.md), so it is kept only as an option.
SPLITS = {
    "object": [(2, 0), (2, 1), (2, 2)],
    "combo": [(0, 0), (1, 1), (2, 2)],
}
DEFAULT_SPLIT = "object"
DEFAULT_HELDOUT = SPLITS[DEFAULT_SPLIT]

OBS_DIM = 19  # ee(3) + gripper width(1) + cubes(3x3) + zones xy(3x2)
STATE_DIM = 4  # SmolVLA observation.state: ee(3) + gripper width(1); no privileged object positions
# Image observations for SmolVLA. v2 images: a scene camera framed to the workspace plus a wrist camera, both at
# 512 px (SmolVLA's input size, so no upsampling). The first pilot used ("phone",) at 256 px; evaluation always
# takes the cameras and size from the training data's features.
IMAGE_CAMERAS = ("scene", "wrist")
IMAGE_SIZE = 512


def task_name(task):
    return f"{CUBES[task[0]]}-{ZONES[task[1]]}"


def parse_task(name):
    c, z = name.split("-")
    return CUBES.index(c), ZONES.index(z)


def resolve_heldout(spec):
    """Held-out task list from a split name ("object", "combo"), "object:<cube>", "zone:<zone>", or a comma
    list of task names like "red-yellow,blue-orange"."""
    if spec in SPLITS:
        return list(SPLITS[spec])
    if spec.startswith("object:"):
        c = CUBES.index(spec.split(":", 1)[1])
        return [(c, z) for z in range(3)]
    if spec.startswith("zone:"):
        z = ZONES.index(spec.split(":", 1)[1])
        return [(c, z) for c in range(3)]
    return [parse_task(s) for s in spec.split(",")]


def instruction(task):
    """Language instruction for SmolVLA."""
    return f"put the {CUBES[task[0]]} cube in the {ZONES[task[1]]} zone"


def task_onehot(task):
    v = np.zeros(6, dtype=np.float32)
    v[task[0]] = 1.0
    v[3 + task[1]] = 1.0
    return v


@dataclass
class EnvConfig:
    control_hz: float = 10.0
    workspace: tuple = (0.0, 0.5, 0.0, 0.35)    # xmin, xmax, ymin, ymax of the marker rectangle
    zone_half: float = 0.05
    cube_half: float = 0.02
    min_zone_sep: float = 0.12                  # Chebyshev distance between zone centers
    min_cube_zone_sep: float = 0.09             # Chebyshev: cubes never start inside a zone
    min_cube_sep: float = 0.09                  # Euclidean distance between cubes
    max_ee_step: float = 0.05                   # max EE target motion per control step (m)
    ee_bounds: tuple = ((-0.05, 0.55), (-0.08, 0.42), (0.008, 0.40))
    home_ee: tuple = (0.45, 0.03, 0.15)         # matches the human HOME spot (near-right corner)
    grip_kp: float = 400.0                      # gripper position gain (menagerie default 100)
    max_steps: int = 200
    # v2 (2026-10-08): fingers close near-far like the human pinch, and the gripper starts closed like the
    # human's relaxed hand. v1 (the Phase 1 mechanism study and the first V0 pilot): yaw 0, start open.
    gripper_yaw_deg: float = 90.0
    start_gripper_closed: bool = True
    release_dist: float = 0.045                 # "released": fingertip center this far from the cube center


LEGACY_V1 = dict(gripper_yaw_deg=0.0, start_gripper_closed=False)


def env_cfg_dict(cfg):
    """The settings that define the data-generating env version (stored with every episode and dataset)."""
    return {"gripper_yaw_deg": float(cfg.gripper_yaw_deg), "start_gripper_closed": bool(cfg.start_gripper_closed)}


def env_config(d=None):
    """EnvConfig from a stored dict (dataset/episode metadata). A missing dict means data made before v2."""
    return EnvConfig(**(LEGACY_V1 if d is None else d))


class PickPlaceEnv:
    def __init__(self, cfg: EnvConfig = None, render_size=(480, 640)):
        self.cfg = cfg or EnvConfig()
        self.m = mujoco.MjModel.from_xml_path(SCENE)
        self.d = mujoco.MjData(self.m)
        self.n_sub = int(round(1.0 / (self.cfg.control_hz * self.m.opt.timestep)))
        # Stiffer gripper: force = kp * (target_width/2 - tendon_len); ctrl in [0, 255] maps to [0, 0.04].
        g = self.m.actuator("actuator8").id
        self.m.actuator_gainprm[g, 0] = 0.04 * self.cfg.grip_kp / 255.0
        self.m.actuator_biasprm[g, 1] = -self.cfg.grip_kp
        self.m.actuator_biasprm[g, 2] = -0.1 * self.cfg.grip_kp
        self.ik = IK(self.m)
        self.ik.rot = down_rot(self.cfg.gripper_yaw_deg)
        self.ee_site = self.m.site("ee").id
        self.cube_qadr = [self.m.jnt_qposadr[self.m.joint(f"cube_{c}").id] for c in CUBES]
        self.cube_body = [self.m.body(f"cube_{c}").id for c in CUBES]
        self.zone_mocap = [self.m.body_mocapid[self.m.body(f"zone_{z}").id] for z in ZONES]
        self.finger_qadr = [self.m.jnt_qposadr[self.m.joint(j).id] for j in ("finger_joint1", "finger_joint2")]
        self.render_size = render_size
        self._renderer = None
        self._img_renderer = None
        # Home arm configuration: menagerie "home" keyframe refined by IK to home_ee, pointing down.
        q_key = self.m.key_qpos[0][:7].copy()
        self.ik.q_rest = q_key.copy()
        q = q_key
        for _ in range(10):
            q = self.ik.solve(q, np.array(self.cfg.home_ee), iters=50)
        self.q_home = q
        self.ik.q_rest = q.copy()
        self.task = None
        self.t = 0

    # ------------------------------------------------------------------ layout
    def sample_layout(self, rng):
        c = self.cfg
        x0, x1, y0, y1 = c.workspace
        for _ in range(10000):
            zh, ch = c.zone_half + 0.01, c.cube_half + 0.02
            zones = np.stack([rng.uniform([x0 + zh, y0 + zh], [x1 - zh, y1 - zh]) for _ in range(3)])
            if any(np.abs(zones[i] - zones[j]).max() < c.min_zone_sep for i in range(3) for j in range(i)):
                continue
            cubes = np.stack([rng.uniform([x0 + ch, y0 + ch], [x1 - ch, y1 - ch]) for _ in range(3)])
            if any(np.linalg.norm(cubes[i] - cubes[j]) < c.min_cube_sep for i in range(3) for j in range(i)):
                continue
            if any(np.abs(cubes[i] - zones[j]).max() < c.min_cube_zone_sep for i in range(3) for j in range(3)):
                continue
            return {"cubes": cubes, "zones": zones}
        raise RuntimeError("could not sample a layout")

    # ------------------------------------------------------------------ core API
    def reset(self, task=(0, 0), seed=None, layout=None, ee_start=None):
        rng = np.random.default_rng(seed)
        self.task = tuple(int(v) for v in task)
        self.layout = layout if layout is not None else self.sample_layout(rng)
        self.layout = {k: np.asarray(v, dtype=float) for k, v in self.layout.items()}
        mujoco.mj_resetData(self.m, self.d)
        q = self.q_home
        if ee_start is not None:
            q = self.ik.solve(q, np.asarray(ee_start, dtype=float), iters=100)
        self.d.qpos[:7] = q
        closed = self.cfg.start_gripper_closed
        self.d.qpos[self.finger_qadr] = 0.0 if closed else 0.04
        self.d.ctrl[:7] = q
        self.d.ctrl[7] = 0.0 if closed else 255.0
        for i, a in enumerate(self.cube_qadr):
            self.d.qpos[a:a + 3] = [*self.layout["cubes"][i], self.cfg.cube_half]
            self.d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        for i, mid in enumerate(self.zone_mocap):
            self.d.mocap_pos[mid] = [*self.layout["zones"][i], 0.0005]
        mujoco.mj_forward(self.m, self.d)
        self.ee_target = self.ee_pos().copy()
        self.t = 0
        return self.obs()

    def step(self, action):
        action = np.asarray(action, dtype=float)
        lo = np.array([b[0] for b in self.cfg.ee_bounds])
        hi = np.array([b[1] for b in self.cfg.ee_bounds])
        tgt = np.clip(action[:3], lo, hi)
        delta = tgt - self.ee_target
        n = np.linalg.norm(delta)
        if n > self.cfg.max_ee_step:
            tgt = self.ee_target + delta * (self.cfg.max_ee_step / n)
        self.ee_target = tgt
        q_cmd = self.ik.solve(self.d.qpos[:7], tgt)
        q_prev = self.d.ctrl[:7].copy()
        self.d.ctrl[7] = 255.0 * (1.0 - float(np.clip(action[3], 0.0, 1.0)))
        for k in range(self.n_sub):  # interpolate joint targets for smooth motion
            self.d.ctrl[:7] = q_prev + (q_cmd - q_prev) * (k + 1) / self.n_sub
            self.d.qfrc_applied[:7] = self.d.qfrc_bias[:7]  # gravity/Coriolis compensation, as on the real arm
            mujoco.mj_step(self.m, self.d)
        self.t += 1
        return self.obs(), self.success()

    # ------------------------------------------------------------------ state
    def ee_pos(self):
        return self.d.site_xpos[self.ee_site]

    def gripper_width(self):
        return float(self.d.qpos[self.finger_qadr].sum())

    def remove_distractors(self):
        """Diagnostic: keep only the task's cube and zone in the scene. The other cubes and zones are moved below the
        table (out of every camera's view; the cubes fall away, nothing collides with them). Call after reset()."""
        c, z = self.task
        for i, a in enumerate(self.cube_qadr):
            if i != c:
                self.d.qpos[a:a + 3] = [0.25 + 0.2 * i, 0.1, -0.5]
        for i, mid in enumerate(self.zone_mocap):
            if i != z:
                self.d.mocap_pos[mid] = [0.25, 0.1, -0.5]
        mujoco.mj_forward(self.m, self.d)
        return self.obs()

    def cube_pos(self, i):
        a = self.cube_qadr[i]
        return self.d.qpos[a:a + 3].copy()

    def obs_dict(self):
        return {
            "ee": self.ee_pos().copy(),
            "gripper": np.array([self.gripper_width()]),
            "cubes": np.stack([self.cube_pos(i) for i in range(3)]),
            "zones": self.layout["zones"].copy(),
        }

    def obs(self):
        o = self.obs_dict()
        return np.concatenate([o["ee"], o["gripper"], o["cubes"].ravel(), o["zones"].ravel()]).astype(np.float32)

    def success(self, task=None):
        c, z = task or self.task
        p = self.cube_pos(c)
        zc = self.layout["zones"][z]
        inside = np.all(np.abs(p[:2] - zc) <= self.cfg.zone_half)
        resting = p[2] < self.cfg.cube_half + 0.005
        # not held: the fingertips are away from the cube (the gripper may be closed again, at rest, like the
        # human's relaxed hand at the end of a demo)
        released = np.linalg.norm(self.ee_pos() - p) > self.cfg.release_dist
        return bool(inside and resting and released)

    def state(self):
        """Robot state for SmolVLA: EE position + gripper width."""
        return np.concatenate([self.ee_pos(), [self.gripper_width()]]).astype(np.float32)

    def images(self, size=IMAGE_SIZE, cameras=IMAGE_CAMERAS):
        """Image observations {camera: uint8 [size, size, 3]} for SmolVLA."""
        if self._img_renderer is None:
            self._img_renderer = mujoco.Renderer(self.m, size, size)
        out = {}
        for cam in cameras:
            self._img_renderer.update_scene(self.d, camera=cam)
            out[cam] = self._img_renderer.render().copy()
        return out

    # ------------------------------------------------------------------ rendering
    def render_camera(self, camera, height, width):
        """Render any camera at any size (renderers are cached per size)."""
        if not hasattr(self, "_sized"):
            self._sized = {}
        key = (height, width)
        if key not in self._sized:
            self._sized[key] = mujoco.Renderer(self.m, height, width)
        r = self._sized[key]
        r.update_scene(self.d, camera=camera)
        return r.render()

    def render(self, camera="top"):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.m, *self.render_size)
        self._renderer.update_scene(self.d, camera=camera)
        return self._renderer.render()

    def close(self):
        for r in (self._renderer, self._img_renderer, *getattr(self, "_sized", {}).values()):
            if r is not None:
                r.close()
        self._sized = {}
        self._renderer = self._img_renderer = None
