"""Inverse launcher: interception workspace W + episode spec sampling.

Every episode's shuttle provably passes through the racket's reachable
workspace with time to react: sample the intercept (p*, t*, v*) first, then
back-integrate the drag ODE to get the launch state (p0, v0), then filter for
plausibility (far-court band, inbound net clearance).

W is a function of arms + mount only; it is computed once (sample joint configs
of each arm, reject self-collisions, FK to that arm's face center, require the
face normal within face_normal_max_deg of net-facing) and cached in
scene/workspace_W.npz. W is the union of the right- and left-arm clouds, so
a sampled p* is reachable by at least one racket.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial import cKDTree

import aero
import mjsim

W_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "scene", "workspace_W.npz")


# ---------------------------------------------------------------------------
# 3a. interception workspace W
# ---------------------------------------------------------------------------

def collision_model(scene_path: str | None = None) -> mujoco.MjModel:
    """Compile the scene with arm/racket self-collision enabled.

    The contype/conaffinity pair filter is baked in at compile time (runtime
    edits don't reach mj_collision), so collision checks need this dedicated
    variant. The static base and the shoulder housings it carries overlap at
    the joint for every joint1 angle (parent-child filtering is off for
    static parents); only those chain-adjacent pairs are excluded. Everything
    else, including racket/wrist against the stand, is a real self-collision.
    """
    spec = mujoco.MjSpec.from_file(scene_path or mjsim.SCENE_PATH)
    for g in spec.geoms:
        owner = g.parent.name if g.parent is not None else ""
        if g.group == 3 and owner.startswith(("arm_", "racket")):
            g.contype = 1
            g.conaffinity = 1
            g.margin = 0.02       # generate near-contacts: dist > 0 measures
        elif g.name == "floor":   # clearance, so planners can require margin
            g.contype = 1
            g.conaffinity = 1
            g.margin = 0.02
    body_names = {spec.body(b.name).name for b in spec.bodies}
    for child in ("arm_link1", "arm_link1L"):
        if child in body_names:
            excl = spec.add_exclude()
            excl.bodyname1 = "arm_base_link"
            excl.bodyname2 = child
    return spec.compile()


def _geom_sides(m: mujoco.MjModel) -> np.ndarray:
    """Per geom: 0 if it rides on the right arm (incl. its racket), 1 on the
    left arm, -1 otherwise (stand, world)."""
    roots = {m.body("arm_link1").id: 0, m.body("arm_link1L").id: 1}
    side = np.full(m.ngeom, -1, dtype=int)
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        while b > 0 and b not in roots:
            b = m.body_parentid[b]
        side[g] = roots.get(b, -1)
    return side


def build_workspace(n_samples: int | None = None, seed: int = 0,
                    save_path: str = W_PATH, verbose: bool = True) -> dict:
    """Sample joint configs, reject self-collisions, FK the face center.

    Each arm is sampled for n_samples configs while the other arm stays at
    its zero pose; contacts involving the idle arm are ignored (it overlaps
    the stand's convex hull in every hanging pose, and the policy can move
    it out of the way; arm-arm clashes are contact pairs at runtime). Keeps configs whose face normal is within
    face_normal_max_deg of the net-facing direction (+y, either face side
    counts) AND that look like a receive posture: racket face in front of
    the chest plane, elbow not behind the back. Saves the point cloud, its
    configs, the arm that reached each point (side: 0 right, 1 left), and a
    conservative shrunk box to save_path.
    """
    p = aero.load_params()
    wp = p["workspace"]
    n = n_samples or wp["n_samples"]
    rng = np.random.default_rng(seed)

    m = collision_model()
    d = mujoco.MjData(m)

    # a throwaway Sim wrapper for the ids (same layout as the runtime scene)
    sim = mjsim.Sim(model=m, data=d, params=p)

    cos_max = np.cos(np.radians(wp["face_normal_max_deg"]))
    depth = wp["self_collision_depth"]
    # natural-receive posture: racket in front of the chest plane, elbow not
    # behind the back (the net is at +y; the base sits at base_y)
    front_y = p["arm"]["base_y"] + wp["face_front_margin"]
    elbow_y_min = p["arm"]["base_y"] - wp["elbow_back_margin"]

    points, configs, normals, sides = [], [], [], []
    n_selfcol = n_orient = n_posture = 0
    arms = [(sim.arm_jids, sim.arm_qadr, sim.face_sid, "arm_link4"),
            (sim.arm_l_jids, sim.arm_l_qadr, sim.face_l_sid, "arm_link4l")]
    geom_side = _geom_sides(m)
    for side, (jids, qadr, face_sid, elbow) in enumerate(arms):
        idle = 1 - side
        lo = m.jnt_range[jids, 0]
        hi = m.jnt_range[jids, 1]
        elbow_bid = m.body(elbow).id
        d.qpos[:] = m.qpos0      # idle arm at zero, shuttle parked
        for i in range(n):
            q = rng.uniform(lo, hi)
            d.qpos[qadr] = q
            mujoco.mj_kinematics(m, d)
            face = d.site_xpos[face_sid]
            if face[2] < wp["min_face_z"]:
                continue
            if face[1] < front_y or d.xpos[elbow_bid][1] < elbow_y_min:
                n_posture += 1
                continue
            normal = d.site_xmat[face_sid].reshape(3, 3)[:, 2]
            if abs(normal[1]) < cos_max:      # either face side may face the net
                n_orient += 1
                continue
            mujoco.mj_collision(m, d)
            col = any(d.contact[c].dist < -depth
                      and geom_side[d.contact[c].geom1] != idle
                      and geom_side[d.contact[c].geom2] != idle
                      for c in range(d.ncon))
            if col:
                n_selfcol += 1
                continue
            points.append(face.copy())
            configs.append(q.copy())
            normals.append(normal.copy())
            sides.append(side)
            if verbose and (i + 1) % 20000 == 0:
                print(f"  side {side}: {i+1}/{n} sampled, {len(points)} kept")

    points = np.array(points)
    configs = np.array(configs)
    normals = np.array(normals)
    sides = np.array(sides, dtype=np.int8)
    margin = wp["margin"]
    box_lo = np.quantile(points, 0.02, axis=0) + margin
    box_hi = np.quantile(points, 0.98, axis=0) - margin
    out = {
        "points": points,
        "configs": configs,
        "normals": normals,
        "sides": sides,
        "box_lo": box_lo,
        "box_hi": box_hi,
    }
    np.savez_compressed(save_path, **out)
    if verbose:
        print(f"W: kept {len(points)}/{2 * n} "
              f"(right {int((sides == 0).sum())}, left {int((sides == 1).sum())}) "
              f"(self-collision {n_selfcol}, orientation {n_orient}, "
              f"posture {n_posture})")
        print(f"   box {np.round(box_lo, 2)} .. {np.round(box_hi, 2)}")
        print(f"   saved {save_path}")
    return out


class Workspace:
    """kd-tree membership test over the cached cloud + conservative box."""

    def __init__(self, path: str = W_PATH):
        data = np.load(path)
        self.points = data["points"]
        self.configs = data["configs"]
        self.normals = data["normals"]
        self.sides = data["sides"]
        self.box_lo = data["box_lo"]
        self.box_hi = data["box_hi"]
        self.tree = cKDTree(self.points)
        self._side_trees: dict = {}
        params = aero.load_params()
        self.radius = params["workspace"]["kdtree_radius"]

    def nearest_configs(self, point, k: int = 5, normal=None,
                        side: int = 0) -> np.ndarray:
        """The stored joint configs of the k cloud points nearest to point,
        nearest first, among the points reached by arm `side` (0 right, the
        arm the scripted baseline drives; 1 left). Every one is collision-free and net-facing by
        construction: use them as IK seeds/attractors to stay on a natural
        branch.

        With normal given, candidates are drawn from a wider position
        neighborhood and ranked by face-normal alignment (either face side)
        first, then distance — an IK descent can close a small position gap
        but rarely escapes a wrong-orientation branch."""
        point = np.asarray(point, dtype=float)
        tree, keep = self._side_tree(side)
        if normal is None:
            _, idx = tree.query(point, k=k)
            return self.configs[keep[np.atleast_1d(idx)]]
        n_des = np.asarray(normal, dtype=float)
        n_des = n_des / np.linalg.norm(n_des)
        dist, idx = tree.query(point, k=min(8 * k, len(keep)))
        idx = keep[np.atleast_1d(idx)]
        dist = np.atleast_1d(dist)
        align = np.abs(self.normals[idx] @ n_des)
        order = np.lexsort((dist, -np.round(align, 1)))
        return self.configs[idx[order[:k]]]

    def _side_tree(self, side: int) -> tuple[cKDTree, np.ndarray]:
        if side not in self._side_trees:
            keep = np.nonzero(self.sides == side)[0]
            self._side_trees[side] = (cKDTree(self.points[keep]), keep)
        return self._side_trees[side]

    def contains(self, p) -> bool:
        p = np.asarray(p, dtype=float)
        if np.any(p < self.box_lo) or np.any(p > self.box_hi):
            return False
        dist, _ = self.tree.query(p)
        return bool(dist <= self.radius)

    def contains_many(self, P) -> np.ndarray:
        P = np.atleast_2d(np.asarray(P, dtype=float))
        inside_box = np.all((P >= self.box_lo) & (P <= self.box_hi), axis=1)
        dist, _ = self.tree.query(P)
        return inside_box & (dist <= self.radius)


def workspace_exists(path: str = W_PATH) -> bool:
    return os.path.exists(path)


def load_workspace(path: str = W_PATH) -> Workspace:
    if not workspace_exists(path):
        raise FileNotFoundError(
            f"{path} missing; run: uv run python -c "
            f"'import launcher; launcher.build_workspace()'")
    return Workspace(path)


# ---------------------------------------------------------------------------
# 3b. episode spec sampling
# ---------------------------------------------------------------------------

@dataclass
class LaunchSpec:
    p0: np.ndarray
    v0: np.ndarray
    p_star: np.ndarray
    t_star: float
    v_star: np.ndarray
    ref_traj: tuple  # (t, P, V) forward-time reference, launch..intercept


def _sweet_weights(w: Workspace, lp: dict, params: dict) -> np.ndarray:
    """Gaussian bias toward the sweet region, restricted to the conservative
    (shrunk-box) core of W so every sampled p* passes the membership test.

    Also requires the pre-hit point p* - prehit_offset * u_out to be in W:
    an intercept whose swing runway starts inside the pedestal or outside
    the workspace is reachable only in contorted poses (the racket unfolds
    late and whiffs; found by watching worst episodes)."""
    sweet = np.array(lp["sweet_point"])
    sigma = np.broadcast_to(np.asarray(lp["sweet_sigma"], dtype=float), (3,))
    d2 = (((w.points - sweet) / sigma) ** 2).sum(axis=1)
    weights = np.exp(-0.5 * d2)
    inside = np.all((w.points >= w.box_lo) & (w.points <= w.box_hi), axis=1)
    u_out, _, _ = aero.solve_u_out(sweet, params["control"]["landing_target"],
                                   params)
    prehit = w.points - params["control"]["prehit_offset"] * u_out
    dist, _ = w.tree.query(prehit)
    inside &= dist <= w.radius
    weights[~inside] = 0.0
    return weights / weights.sum()


def sample_specs(n: int, seed: int | None = None, params: dict | None = None,
                 workspace: Workspace | None = None,
                 max_batches: int = 200, batch: int = 2048) -> list[LaunchSpec]:
    """Sample n hittable-by-construction episode specs.

    Rejection sampling runs in numpy batches: the whole candidate batch is
    back-integrated simultaneously (rk4_step is shape-(B,3) capable), with
    per-candidate divergence bail-out. Acceptance is ~1%: most draws imply an
    unphysical launch speed (anti-damped backward drag), the rest miss the
    launch band or the net clearance.
    """
    p = params or aero.load_params()
    lp = p["launcher"]
    k = p["shuttle"]["k"]
    dt = p["integrator"]["dt"]
    net_z = p["net"]["height_top"] + lp["net_margin"]
    w = workspace or load_workspace()
    rng = np.random.default_rng(seed)
    weights = _sweet_weights(w, lp, p)
    cap2 = float(lp["max_launch_speed"]) ** 2

    specs = []
    for _ in range(max_batches):
        if len(specs) >= n:
            break
        idx = rng.choice(len(w.points), size=batch, p=weights)
        p_star = w.points[idx]
        t_star = rng.uniform(*lp["t_star_range"], size=batch)
        speed = rng.uniform(*lp["speed_range"], size=batch)
        descent = np.radians(rng.uniform(*lp["descent_deg_range"], size=batch))
        yaw = np.radians(rng.uniform(-lp["yaw_deg_max"], lp["yaw_deg_max"],
                                     size=batch))
        # incoming: descending, arriving from over the net (moving toward -y)
        v_star = np.stack([
            -speed * np.cos(descent) * np.sin(yaw),
            -speed * np.cos(descent) * np.cos(yaw),
            -speed * np.sin(descent),
        ], axis=1)

        # batched backward integration with per-candidate stopping
        n_steps = np.round(t_star / dt).astype(int)
        pos = p_star.copy()
        vel = v_star.copy()
        alive = np.ones(batch, dtype=bool)
        p0 = np.full((batch, 3), np.nan)
        v0 = np.full((batch, 3), np.nan)
        for s in range(1, n_steps.max() + 1):
            pos[alive], vel[alive] = aero.rk4_step(pos[alive], vel[alive],
                                                   k, -dt)
            diverged = alive & ((vel * vel).sum(axis=1) > cap2)
            alive &= ~diverged
            done = alive & (n_steps == s)
            p0[done] = pos[done]
            v0[done] = vel[done]
            alive &= ~done

        ok = np.all(np.isfinite(p0), axis=1)
        ok &= (p0[:, 1] >= lp["launch_y_range"][0]) & \
              (p0[:, 1] <= lp["launch_y_range"][1])
        ok &= (p0[:, 2] >= lp["launch_z_range"][0]) & \
              (p0[:, 2] <= lp["launch_z_range"][1])
        ok &= np.abs(p0[:, 0]) <= lp["launch_x_max"]

        for i in np.nonzero(ok)[0]:
            if len(specs) >= n:
                break
            # reference trajectory: forward roll from the recovered launch
            t, P, V = aero.simulate(p0[i], v0[i], k, dt=dt,
                                    t_max=t_star[i] + dt / 2,
                                    stop_at_ground=False)
            if not _clears_net(P, net_z):
                continue
            specs.append(LaunchSpec(p0[i].copy(), v0[i].copy(),
                                    p_star[i].copy(), float(t_star[i]),
                                    v_star[i].copy(), (t, P, V)))
    if len(specs) < n:
        raise RuntimeError(
            f"launcher: only {len(specs)}/{n} specs after {max_batches} "
            f"batches; check W / launcher params")
    return specs


def _clears_net(P: np.ndarray, net_z: float) -> bool:
    """True if the trajectory crosses y=0 exactly once, above net_z."""
    y = P[:, 1]
    sign_change = np.nonzero(np.diff(np.sign(y)) != 0)[0]
    if len(sign_change) != 1:
        return False
    i = sign_change[0]
    frac = y[i] / (y[i] - y[i + 1])
    z_cross = P[i, 2] + frac * (P[i + 1, 2] - P[i, 2])
    return bool(z_cross >= net_z)


def save_specs(specs: list[LaunchSpec], path: str) -> None:
    np.savez_compressed(
        path,
        p0=np.array([s.p0 for s in specs]),
        v0=np.array([s.v0 for s in specs]),
        p_star=np.array([s.p_star for s in specs]),
        t_star=np.array([s.t_star for s in specs]),
        v_star=np.array([s.v_star for s in specs]),
    )


def load_specs(path: str, with_traj: bool = True) -> list[LaunchSpec]:
    """Load saved specs; reference trajectories are re-integrated on load."""
    data = np.load(path)
    params = aero.load_params()
    k = params["shuttle"]["k"]
    dt = params["integrator"]["dt"]
    specs = []
    for i in range(len(data["t_star"])):
        traj = None
        if with_traj:
            t, P, V = aero.simulate(data["p0"][i], data["v0"][i], k, dt=dt,
                                    t_max=data["t_star"][i] + dt / 2,
                                    stop_at_ground=False)
            traj = (t, P, V)
        specs.append(LaunchSpec(data["p0"][i], data["v0"][i],
                                data["p_star"][i], float(data["t_star"][i]),
                                data["v_star"][i], traj))
    return specs


if __name__ == "__main__":
    build_workspace()
