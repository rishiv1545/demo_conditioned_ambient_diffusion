"""Damped least-squares IK on a site's position + orientation (arm joints only)."""
import mujoco
import numpy as np


def down_rot(yaw_deg=90.0):
    """Gripper pointing straight down; the fingers open along (cos yaw, sin yaw, 0) in the table frame.
    yaw 0: along x (left-right as seen from the human's seat); yaw 90: along y (near-far, like the human pinch).
    Columns are the site's x, y, z axes in the world frame (site y = finger axis, site z = approach direction)."""
    a = np.radians(yaw_deg)
    y = np.array([np.cos(a), np.sin(a), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([np.cross(y, z), y, z])


DOWN_ROT = down_rot(0.0)  # the original (v1) orientation


class IK:
    def __init__(self, model, site="ee", n_arm=7, damping=0.05, null_gain=0.1):
        self.m = model
        self.d = mujoco.MjData(model)          # scratch data; never stepped
        self.site = model.site(site).id
        self.n = n_arm                         # arm joints are the first n qpos/dof entries
        self.lam2 = damping ** 2
        self.null_gain = null_gain
        self.lo = model.jnt_range[:n_arm, 0].copy()
        self.hi = model.jnt_range[:n_arm, 1].copy()
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.q_rest = None                     # nullspace posture bias, set by the env
        self.rot = DOWN_ROT                    # default target orientation, set by the env

    def fk(self, q):
        self.d.qpos[:self.n] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, q0, target_pos, target_rot=None, iters=15, tol=1e-4):
        target_rot = self.rot if target_rot is None else target_rot
        q = np.array(q0[:self.n], dtype=float)
        for _ in range(iters):
            pos, rot = self.fk(q)
            e_pos = target_pos - pos
            e_rot = 0.5 * sum(np.cross(rot[:, i], target_rot[:, i]) for i in range(3))
            e = np.concatenate([e_pos, e_rot])
            if np.linalg.norm(e) < tol:
                break
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.site)
            J = np.vstack([self.jacp[:, :self.n], self.jacr[:, :self.n]])
            JJt_inv = np.linalg.inv(J @ J.T + self.lam2 * np.eye(6))
            J_pinv = J.T @ JJt_inv
            dq = J_pinv @ e
            if self.q_rest is not None:
                dq += (np.eye(self.n) - J_pinv @ J) @ (self.null_gain * (self.q_rest - q))
            q = np.clip(q + dq, self.lo, self.hi)
        return q
