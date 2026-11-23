"""Optional gait, camera, and reset diagnostics for ``play.py``.

Enable with ``play.py --diagnostics``. The printed formats (``[PLAYARC]``,
``[PLAYLEAD]``, ...) are parsed by ``tools/onshape_course/probe_report.py``, so
keep them stable.

The behaviour is tuned for the Onshape courses and is driven by environment
variables so a probe does not need extra CLI plumbing:

    PLAY_INIT_OVERRIDE   joint overrides applied to the spawn pose
    PLAY_SPAWN_DY        per-env spawn offset along the course axis
    PLAY_MARKER          world x,y,z of a visible spawn-marker sphere
    PLAY_FOLLOW_ROBOT    keep robot 0 in frame (any value except ""/"0")
    PLAY_REAR_CAM        rear follow view (implies PLAY_FOLLOW_ROBOT)
    PLAY_CAM_MODE        side (default), rear, or quarter
    PLAY_CAM_ALPHA/DIST/HEIGHT/LOOK   follow-camera geometry
    PLAY_PUSH_DETECT     report root-velocity jumps (default on)
"""

from __future__ import annotations

import math
import os

import torch


def _foot_up_in_body(quat) -> tuple:
    """World +Z expressed in a body's own frame -- "which way is up" for that body.

    Convention-free: for a foot resting flat on level ground this is a fixed
    vector of the foot's local frame, whatever that frame's axes happen to be,
    so the angle between it and a flat-ground reference *is* the landing tilt
    without needing to know which local axis is the sole normal.
    """
    w, x, y, z = (float(v) for v in quat)
    return (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y))


def _swing_arc_step(env, state: dict) -> None:
    """Per-stride swing diagnostics for robot 0, tracked per foot.

    One ``[PLAYARC]`` line per swing, printed on touchdown, with:
      dy_min   most-rearward excursion behind the root (good step ~ -0.2;
               a big negative is the rear "butt-kick")
      dz_max   peak lift above the root
      dx_lift  lateral offset of the foot at peak lift  -- this is the number
               that exposes a foot swinging out to the side (circumduction)
      dx_max   largest |lateral offset| over the whole swing -- the calibration
               input for the foot_lateral_splay deadband
      dx_td    lateral offset at touchdown
    plus the swing leg's hip ab/adduction, hip roll, hip pitch and knee at peak
    lift. Lateral offsets are world x: the robot faces +Y up this course, so
    world y is fore/aft and world x *is* the lateral axis. Foot_L/Foot_R are the
    right/left legs, so a right-leg circumduction shows up as dx_lift growing
    well beyond the ~+0.15 m stance half-width.
    """
    try:
        rob = env.unwrapped.scene["robot"]
        bodies = list(rob.data.body_names)
        joints = list(rob.data.joint_names)
        root = rob.data.root_pos_w[0]
        legs = state.setdefault("legs", {})
        # Stance-phase tilt tracker: does the foot stay off-flat for the whole
        # stance, or land badly and settle? Also tells us whether a
        # whole-contact reward would charge a normal toe-off.
        stance = state.setdefault("stance", {})
        for side, sensor in (("L", "left_foot_terrain_contact"), ("R", "right_foot_terrain_contact")):
            st = stance.setdefault(side, {"frames": 0, "max": 0.0, "td": 0.0})
            air = env.unwrapped.scene[sensor].data.current_air_time[0].item()
            f_up = _foot_up_in_body(rob.data.body_quat_w[0, bodies.index(f"Foot_{side}")])
            tilt_now = math.degrees(math.acos(max(-1.0, min(1.0, f_up[2]))))
            if air <= 0.05:
                if st["frames"] == 0:
                    st["td"] = tilt_now
                st["frames"] += 1
                st["max"] = max(st["max"], tilt_now)
            elif st["frames"] > 0:
                print(f"[PLAYARCSTANCE] {side} td={st['td']:5.1f}deg max={st['max']:5.1f}deg "
                      f"frames={st['frames']}", flush=True)
                st.update(frames=0, max=0.0, td=0.0)
        for side, sensor in (("L", "left_foot_terrain_contact"), ("R", "right_foot_terrain_contact")):
            leg = legs.setdefault(side, {"active": False, "min_dy": 0.0, "max_dz": -9.0,
                                         "dx_lift": 0.0, "dx_td": 0.0, "dx_max": 0.0,
                                         "roll": 0.0, "stance_dx": 0.0, "joints": {},
                                         "knee_z": 0.0, "root_y": 0.0})
            foot = rob.data.body_pos_w[0, bodies.index(f"Foot_{side}")]
            air = env.unwrapped.scene[sensor].data.current_air_time[0].item()
            dx = (foot[0] - root[0]).item()
            dy = (foot[1] - root[1]).item()
            dz = (foot[2] - root[2]).item()
            if air > 0.05:
                if not leg["active"]:
                    leg.update(active=True, min_dy=0.0, max_dz=-9.0, dx_lift=dx, dx_td=dx, dx_max=abs(dx))
                leg["dx_max"] = max(leg["dx_max"], abs(dx))
                leg["min_dy"] = min(leg["min_dy"], dy)
                if dz > leg["max_dz"]:
                    leg["max_dz"] = dz
                    leg["dx_lift"] = dx
                    # Knee height above the root at peak lift: the calf link's
                    # frame sits at the knee joint, so this is where the knee
                    # actually is. The visual eval's "raises the knee too high"
                    # is this number (chest height is ~1.0 m on this robot).
                    knee_body = "Calf_L" if side == "L" else "Calf_R"
                    leg["knee_z"] = float(
                        rob.data.body_pos_w[0, bodies.index(knee_body)][2] - root[2]
                    )
                    leg["root_y"] = float(root[1])
                    # Real joint set (from joint_names above): Hip_F is the hip
                    # pitch, Hip_A ab/adduction, Hip_R the hip roll.
                    leg["joints"] = {
                        name: rob.data.joint_pos[0, joints.index(f"{name}_{side}")].item()
                        for name in ("Hip_F", "Hip_A", "Hip_R", "Knee", "Ankle_P", "Ankle_R")
                        if f"{name}_{side}" in joints
                    }
                    # Mechanism diagnostics at peak lift: torso roll (projected
                    # gravity Y) and the *stance* foot's lateral offset, so a
                    # foot-space splay can be attributed to the swing leg, a
                    # leaning torso, or a sideways weight shift.
                    leg["roll"] = float(rob.data.projected_gravity_b[0, 1])
                    other = rob.data.body_pos_w[0, bodies.index("Foot_L" if side == "R" else "Foot_R")]
                    leg["stance_dx"] = float(other[0] - root[0])
                leg["dx_td"] = dx
            elif leg["active"]:
                j = leg["joints"]
                # Touchdown framing. The first touchdown on the flat approach
                # (root y < 0.3, ahead of the first flight at y = 0) defines that
                # foot's own "flat" reference; every later touchdown is reported
                # as the angle away from it, plus the ankle joints. This is the
                # calibration input for a contact-gated flat-foot reward term.
                foot_i = bodies.index(f"Foot_{side}")
                up = _foot_up_in_body(rob.data.body_quat_w[0, foot_i])
                refs = state.setdefault("flat_ref", {})
                on_flat = float(root[1]) < 0.3
                if side not in refs and on_flat:
                    refs[side] = up
                ref = refs.get(side)
                if ref is not None:
                    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(up, ref))))
                    abs_tilt = math.degrees(math.acos(max(-1.0, min(1.0, up[2]))))
                    ank = {name: rob.data.joint_pos[0, joints.index(f"{name}_{side}")].item()
                           for name in ("Ankle_P", "Ankle_R") if f"{name}_{side}" in joints}
                    print(f"[PLAYARCFOOT] {side} tilt={math.degrees(math.acos(dot)):5.1f}deg "
                          f"abs={abs_tilt:5.1f}deg flat_ref={on_flat} "
                          f"up=({up[0]:+.3f},{up[1]:+.3f},{up[2]:+.3f}) "
                          f"AnkleP={ank.get('Ankle_P', float('nan')):+.3f} "
                          f"AnkleR={ank.get('Ankle_R', float('nan')):+.3f}", flush=True)
                print(f"[PLAYARC] {side} dy_min={leg['min_dy']:+.3f} dz_max={leg['max_dz']:.3f} "
                      f"dx_lift={leg['dx_lift']:+.3f} dx_td={leg['dx_td']:+.3f} "
                      f"dx_max={leg['dx_max']:.3f} "
                      f"HipF={j.get('Hip_F', float('nan')):+.3f} HipA={j.get('Hip_A', float('nan')):+.3f} "
                      f"HipR={j.get('Hip_R', float('nan')):+.3f} Knee={j.get('Knee', float('nan')):+.3f} "
                      f"AnkP={j.get('Ankle_P', float('nan')):+.3f} AnkR={j.get('Ankle_R', float('nan')):+.3f} "
                      f"roll={leg['roll']:+.3f} stance_dx={leg['stance_dx']:+.3f} "
                      f"knee_z={leg.get('knee_z', float('nan')):+.3f} "
                      f"root_y={leg.get('root_y', float('nan')):+.2f}",
                      flush=True)
                leg.update(active=False, min_dy=0.0, max_dz=-9.0, dx_lift=0.0, dx_td=0.0,
                           dx_max=0.0, roll=0.0, stance_dx=0.0, joints={},
                           knee_z=0.0, root_y=0.0)
    except Exception as err:  # noqa: BLE001 - diagnostic only
        state["err"] = str(err)
        print(f"[PLAYARC] disabled: {err}", flush=True)


def _follow_cam_step(
    env,
    state: dict,
    mode: str = "side",
    alpha: float = 0.05,
    dist: float = 3.4,
    height: float = 1.15,
    look: float = 1.2,
) -> None:
    """Track the robot with a low-pass filter, so the shot does not judder.

    The pelvis bobs and pitches with every step, and the robot's speed is not
    constant on the course, so a camera that simply copies the root's position
    makes the whole frame shake (the old follow-cam snapped to the root every
    10 steps, which is worse still: the jump is visible as a jolt). Here the
    desired eye/target are computed from the root and then low-pass filtered
    with ``alpha`` (the fraction of the remaining error closed per step, so
    alpha = 0.05 is a ~0.4 s time constant at 50 Hz).

    The height follows the robot, so the camera rises smoothly as it climbs;
    the look-at point sits ahead of the robot so it walks *into* the frame
    rather than sliding out of it.
    """
    try:
        p = env.unwrapped.scene["robot"].data.root_pos_w[0].tolist()
        if mode == "rear":
            # Behind and slightly to the side, looking up the course: the
            # fore/aft axis is screen-vertical, so the stride reads directly.
            eye = [p[0] + 0.25 * dist * 0.35, p[1] - dist, p[2] + height]
            target = [p[0], p[1] + look, p[2] + 0.05]
        elif mode == "quarter":
            # Front three-quarter: shows the knee lift and the foot placement
            # at the same time, at the cost of some fore/aft readability.
            eye = [p[0] + 0.75 * dist, p[1] - 0.6 * dist, p[2] + height]
            target = [p[0], p[1] + 0.45 * look, p[2] - 0.05]
        else:  # side
            eye = [p[0] + dist, p[1] - 0.25 * look, p[2] + height]
            target = [p[0], p[1] + 0.35 * look, p[2] - 0.10]
        if state["eye"] is None:
            state["eye"], state["target"] = list(eye), list(target)
        else:
            state["eye"] = [a + alpha * (b - a) for a, b in zip(state["eye"], eye)]
            state["target"] = [a + alpha * (b - a) for a, b in zip(state["target"], target)]
        env.unwrapped.sim.set_camera_view(eye=state["eye"], target=state["target"])
        if state.get("announced") is None:
            state["announced"] = True
            print(f"[PLAYCAM] mode={mode} eye={[round(v, 2) for v in state['eye']]} "
                  f"target={[round(v, 2) for v in state['target']]}", flush=True)
    except Exception as err:  # noqa: BLE001 - headless has no viewport
        print(f"[PLAYCAM] camera failed: {err}", flush=True)


def _lead_step(env, state: dict) -> None:
    """Which foot first steps *up* onto the course, per env -- the lead-foot test.

    This Onshape env is deterministic (fixed spawn, fixed command), so the robot
    always meets the first riser at the same gait phase and therefore always
    mounts with the same foot. ``PLAY_SPAWN_DY`` steps the spawn along the
    approach to sweep that arrival phase; this reports, per env, which foot made
    the first raised touchdown, how many flat-ground plants preceded it (the
    "shuffle" it burns to re-time), and the sequence of those plants.

    A "lead" plant is the first foot plant whose height exceeds the flat-foot
    reference by 0.07 m: the first riser is 0.14 m, so that plant is on the
    step. Foot height is the right signal here (not foot_y, which lags because
    the foot body centre trails the sole contact as it lands on the edge, and
    not root distance, a poor proxy for where the foot is).
    """
    try:
        rob = env.unwrapped.scene["robot"]
        bodies = list(rob.data.body_names)
        root = rob.data.root_pos_w
        envs = state.setdefault("envs", {})
        contacts = state.setdefault("contact", {})
        for side, sensor in (("L", "left_foot_terrain_contact"), ("R", "right_foot_terrain_contact")):
            air = env.unwrapped.scene[sensor].data.current_air_time[:, 0]
            contact = air <= 0.05
            foot = rob.data.body_pos_w[:, bodies.index(f"Foot_{side}")]
            prev = contacts.get(side)
            contacts[side] = contact.clone()
            if prev is None:
                continue
            for i in torch.nonzero(contact & ~prev).flatten().tolist():
                st = envs.setdefault(i, {"ground": None, "lead": None, "steps": 0,
                                         "seq": [], "max_z": -9.0})
                fz, fy = float(foot[i, 2]), float(foot[i, 1])
                st["max_z"] = max(st["max_z"], fz)
                if st["lead"] is not None:
                    continue
                if st["ground"] is None:
                    # First plant of the episode establishes the flat-foot height
                    # (all spawns sit on the flat approach, several metres short
                    # of the y=0 riser base).
                    st["ground"] = fz
                    continue
                st["ground"] = min(st["ground"], fz)
                # A foot planted on the first step sits ~0.1 m above the flat-foot
                # height (the first riser is 0.14 m). 0.07 m clears every flat
                # plant measured (0.045-0.076) with margin yet catches the step.
                if fz > st["ground"] + 0.07:
                    st["lead"] = side
                    print(f"[PLAYLEAD] env={i} LEAD={side} shuffle_steps={st['steps']} "
                          f"seq={''.join(st['seq'])} foot=(y{fy:+.3f} z{fz:+.3f}) "
                          f"flat_z={st['ground']:+.3f} root_y={float(root[i, 1]):+.2f}", flush=True)
                else:
                    st["steps"] += 1
                    st["seq"].append(side)
                    print(f"[PLAYLEAD] env={i} plant={side} n={st['steps']} "
                          f"foot=(y{fy:+.3f} z{fz:+.3f}) root_y={float(root[i, 1]):+.2f}", flush=True)
    except Exception as err:  # noqa: BLE001 - diagnostic only
        state["err"] = str(err)
        print(f"[PLAYLEAD] disabled: {err}", flush=True)


def apply_init_override(env_cfg) -> None:
    """Override the spawn joint pose from ``PLAY_INIT_OVERRIDE``.

    Format: ``"Ankle_P_R=-0.10,Hip_A_R=-0.04"``. Leaves the shipped config
    untouched, so a symmetry experiment does not require editing it.
    """
    spec = os.environ.get("PLAY_INIT_OVERRIDE", "").strip()
    if not spec:
        return
    joint_pos = env_cfg.scene.robot.init_state.joint_pos
    for entry in spec.split(","):
        if not entry.strip():
            continue
        key, value = entry.split("=")
        key = key.strip()
        old = joint_pos.get(key)
        joint_pos[key] = float(value)
        print(f"[PLAYGEO] init override {key}: {old} -> {joint_pos[key]}", flush=True)


def apply_spawn_offsets(env) -> None:
    """Shift each env's spawn along the course from ``PLAY_SPAWN_DY``.

    ``PLAY_SPAWN_DY`` is a comma-separated list, one value per env (repeats if
    fewer than num_envs). ``reset_root_state_uniform`` adds ``env_origins`` to
    the sampled spawn on every reset, so shifting the per-env origin y is a
    persistent spawn offset with no change to the reset event.
    """
    spec = os.environ.get("PLAY_SPAWN_DY", "").strip()
    if not spec:
        return
    offsets = [float(value) for value in spec.split(",") if value.strip()]
    origins = env.unwrapped.scene.env_origins
    for index in range(origins.shape[0]):
        origins[index, 1] += offsets[index % len(offsets)]
    print(f"[PLAYGEO] PLAY_SPAWN_DY={spec} -> env_origins="
          f"{[[round(v, 2) for v in row.tolist()] for row in origins]}", flush=True)


def pin_course_camera(env) -> None:
    """Point the viewport at the Onshape course before playback starts."""
    # The course spans +Y; robots start staggered across the 10 m width near
    # the marker. Also applies under --video: rgb_array frames are rendered
    # from this same viewport camera.
    try:
        env.unwrapped.sim.set_camera_view(
            eye=(2.0, -6.0, 2.8), target=(4.4, 0.5, 0.8))
        from isaacsim.core.utils.viewports import set_active_viewport_camera
        set_active_viewport_camera("/OmniverseKit_Persp")
        print("[PLAY] camera pinned on Onshape course", flush=True)
    except Exception as err:  # noqa: BLE001 - headless has no viewport
        print(f"[PLAY] camera pin skipped: {err}", flush=True)


def add_spawn_marker(env) -> None:
    """Draw a visible sphere at ``PLAY_MARKER`` (world x,y,z).

    The Onshape marker sliver itself is excluded from the merged terrain (it is
    a spawn pad without collision), so it is invisible in the render and there
    is nothing to check the spawn against.
    """
    spec = os.environ.get("PLAY_MARKER", "").strip()
    if not spec:
        return
    try:
        x, y, z = (float(value) for value in spec.split(","))
        from pxr import Gf, UsdGeom

        stage = env.unwrapped.sim.stage
        sphere = UsdGeom.Sphere.Define(stage, "/World/spawn_marker")
        sphere.CreateRadiusAttr(0.09)
        UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(x, y, z + 0.06))
        print(f"[PLAYMARK] marker ball at ({x:.3f}, {y:.3f}, {z:.3f})", flush=True)
    except Exception as err:  # noqa: BLE001 - diagnostic only
        print(f"[PLAYMARK] failed: {err}", flush=True)


class PlayDiagnostics:
    """Per-step gait, camera, reward, and reset instrumentation for ``play.py``."""

    def __init__(self, env):
        self.env = env
        self.dt = env.unwrapped.step_dt
        self.swing = {"legs": {}, "err": None}
        self.lead = {}
        self.cam = {"eye": None, "target": None}
        self.push_state = {"v": None}
        self.ep_names = None
        self.ep_sum = None
        self.geo_done = False

        follow = os.environ.get("PLAY_FOLLOW_ROBOT", "") not in ("", "0")
        rear = os.environ.get("PLAY_REAR_CAM", "") not in ("", "0")
        self.follow_cam = follow or rear
        self.cam_mode = os.environ.get("PLAY_CAM_MODE", "rear" if rear else "side").strip().lower()
        self.cam_alpha = float(os.environ.get("PLAY_CAM_ALPHA", "0.05") or 0.05)
        self.cam_dist = float(os.environ.get("PLAY_CAM_DIST", "3.4") or 3.4)
        self.cam_height = float(os.environ.get("PLAY_CAM_HEIGHT", "0.55") or 0.55)
        self.cam_look = float(os.environ.get("PLAY_CAM_LOOK", "1.2") or 1.2)
        self.push_detect = os.environ.get("PLAY_PUSH_DETECT", "1") not in ("", "0")

        print(f"[PLAYARC] joint_names={list(env.unwrapped.scene['robot'].data.joint_names)}", flush=True)
        print(f"[PLAYARC] body_names={list(env.unwrapped.scene['robot'].data.body_names)}", flush=True)

    def on_step(self, actions, dones, step: int) -> None:
        """Run every enabled diagnostic after one environment step."""
        self._spawn_geometry(step)
        self._rewards(dones, step)
        if self.follow_cam:
            _follow_cam_step(
                self.env, self.cam, mode=self.cam_mode, alpha=self.cam_alpha,
                dist=self.cam_dist, height=self.cam_height, look=self.cam_look,
            )
        self._push(dones, step)
        if not self.swing["err"]:
            _swing_arc_step(self.env, self.swing)
        if not self.lead.get("err"):
            _lead_step(self.env, self.lead)
        if step % 200 == 0:
            self._debug(actions, dones, step)

    def _spawn_geometry(self, step: int) -> None:
        if self.geo_done or step < 2:
            return
        self.geo_done = True
        try:
            rob = self.env.unwrapped.scene["robot"]
            names = list(rob.data.body_names)
            root = rob.data.root_pos_w[0]
            foot_l = (rob.data.body_pos_w[0, names.index("Foot_L")] - root).tolist()
            foot_r = (rob.data.body_pos_w[0, names.index("Foot_R")] - root).tolist()
            print(f"[PLAYGEO] spawn foot_L=({foot_l[0]:+.3f},{foot_l[1]:+.3f},{foot_l[2]:+.3f}) "
                  f"foot_R=({foot_r[0]:+.3f},{foot_r[1]:+.3f},{foot_r[2]:+.3f}) "
                  f"root=({float(root[0]):+.3f},{float(root[1]):+.3f},{float(root[2]):+.3f}) "
                  f"heading={float(rob.data.heading_w[0]):+.3f} "
                  f"pg=({float(rob.data.projected_gravity_b[0,0]):+.3f},"
                  f"{float(rob.data.projected_gravity_b[0,1]):+.3f},"
                  f"{float(rob.data.projected_gravity_b[0,2]):+.3f})", flush=True)
        except Exception as err:  # noqa: BLE001 - diagnostic only
            print(f"[PLAYGEO] failed: {err}", flush=True)
        # Direct symmetry audit at spawn, in world coordinates (no link frame
        # conventions involved): for each left/right body pair, report the
        # lateral midpoint offset from the root -- zero for a mirror-symmetric
        # robot at a symmetric pose -- and the orientation residual against the
        # mirrored right body (quaternion mirror about the world x=const plane
        # is (w,-x,y,z)).
        try:
            rob = self.env.unwrapped.scene["robot"]
            names = list(rob.data.body_names)
            root = rob.data.root_pos_w[0]
            print("[PLAYMIRROR] pair    mid_x-root      dpos_y   dpos_z    d_quat", flush=True)
            for name in ("Hip_F", "Hip_A", "Hip_R", "Calf", "Ankle", "Foot"):
                left = names.index(name + "_L") if name + "_L" in names else None
                right = names.index(name + "_R") if name + "_R" in names else None
                if left is None or right is None:
                    continue
                pos_l = rob.data.body_pos_w[0, left]
                pos_r = rob.data.body_pos_w[0, right]
                mid = 0.5 * float(pos_l[0] + pos_r[0]) - float(root[0])
                dy = float(pos_l[1] - pos_r[1])
                dz = float(pos_l[2] - pos_r[2])
                quat_l = rob.data.body_quat_w[0, left]
                quat_r = rob.data.body_quat_w[0, right]
                mirrored = [quat_r[0], -quat_r[1], quat_r[2], quat_r[3]]
                dq = min(max(abs(float(quat_l[k]) - float(mirrored[k])) for k in range(4)),
                         max(abs(float(quat_l[k]) + float(mirrored[k])) for k in range(4)))
                print(f"[PLAYMIRROR] {name:7} {mid:+8.4f}  {dy:+8.4f} {dz:+8.4f}  {dq:8.4f}", flush=True)
            print(f"[PLAYMIRROR] root=({float(root[0]):+.4f},{float(root[1]):+.4f},{float(root[2]):+.4f}) "
                  f"-> a symmetric spawn needs mid_x-root = 0 and d_quat ~ 0 for every pair",
                  flush=True)
        except Exception as err:  # noqa: BLE001 - diagnostic only
            print(f"[PLAYMIRROR] failed: {err}", flush=True)

    def _rewards(self, dones, step: int) -> None:
        rm = self.env.unwrapped.reward_manager
        if self.ep_names is None:
            self.ep_names = list(getattr(rm, "_term_names",
                                         [f"term_{i}" for i in range(rm._step_reward.shape[1])]))
            self.ep_sum = torch.zeros(len(self.ep_names), dtype=torch.float64, device=rm._step_reward.device)
        # Accumulated across the episode for robot 0: shows which terms actually
        # drive this policy on this course (the input to reward tuning).
        self.ep_sum += rm._step_reward[0].to(torch.float64)
        # Catch the exact reset: the 200-step debug schedule below lands after
        # the term flags have cleared, so a falling policy just shows up as a
        # shrinking episode counter with no cause.
        if not dones.any():
            return
        tm = self.env.unwrapped.termination_manager
        ry = self.env.unwrapped.scene["robot"].data.root_pos_w[:, 1].tolist()
        rz = self.env.unwrapped.scene["robot"].data.root_pos_w[:, 2].tolist()
        rx = self.env.unwrapped.scene["robot"].data.root_pos_w[:, 0].tolist()
        # On a spiral the polar coordinates are the meaningful ones: the radius
        # says how far off the walking line the robot is, the angle says where
        # it is around the staircase.
        polar = [f"r={math.hypot(x, y):.2f}@{math.degrees(math.atan2(y, x)):.0f}deg"
                 for x, y in zip(rx, ry)]
        print(f"[PLAYRESET] step={step} root_y={[round(v, 2) for v in ry]} "
              f"root_z={[round(v, 2) for v in rz]} polar={polar} "
              f"terms={tm.active_terms} "
              f"dones={tm._term_dones.tolist()}", flush=True)
        ranked = torch.argsort(self.ep_sum.abs(), descending=True)
        print("[PLAYREW] " + " ".join(
            f"{self.ep_names[i]}={self.ep_sum[i].item():+.1f}" for i in ranked[:14]), flush=True)
        # The reward manager stores the *weighted* step reward, so a sign
        # question about a term needs its raw value. Call the path-relative
        # terms directly.
        try:
            raw = []
            for index, name in enumerate(rm._term_names):
                if not name.startswith("spiral"):
                    continue
                cfg = rm._term_cfgs[index]
                value = cfg.func(self.env.unwrapped, **cfg.params)
                weighted = rm._step_reward[0, index].item()
                raw.append(f"{name}: raw={value[0].item():+.4f} weighted={weighted:+.5f}")
            if raw:
                print("[PLAYRAW] " + " | ".join(raw), flush=True)
        except Exception as err:  # noqa: BLE001 - diagnostic only
            print(f"[PLAYRAW] failed: {err}", flush=True)
        self.ep_sum.zero_()
        # Record each finished env's lead-foot result before clearing it.
        finished = self.lead.get("envs", {})
        for index in torch.nonzero(dones).flatten().tolist():
            state = finished.get(index)
            if state is not None:
                print(f"[PLAYLEAD] env={index} episode_end lead={state['lead']} "
                      f"shuffle_steps={state['steps']} seq={''.join(state['seq'])} "
                      f"max_foot_z={state['max_z']:+.3f}", flush=True)
                finished.pop(index, None)

    def _push(self, dones, step: int) -> None:
        """Report root-velocity jumps: the push event sets the velocity in one step."""
        if not self.push_detect:
            return
        velocity = self.env.unwrapped.scene["robot"].data.root_lin_vel_w[0].tolist()
        previous = self.push_state.get("v")
        if previous is not None and not bool(dones.any()):
            dvx, dvy = velocity[0] - previous[0], velocity[1] - previous[1]
            if max(abs(dvx), abs(dvy)) > 0.25:
                root_y = float(self.env.unwrapped.scene["robot"].data.root_pos_w[0][1])
                print(f"[PLAYPUSH] step={step} t={step * self.dt:.2f}s "
                      f"dv=({dvx:+.3f},{dvy:+.3f}) v=({velocity[0]:+.2f},{velocity[1]:+.2f}) "
                      f"root_y={root_y:+.2f}", flush=True)
        self.push_state["v"] = velocity

    def _debug(self, actions, dones, step: int) -> None:
        scene = self.env.unwrapped.scene
        rx = scene["robot"].data.root_pos_w[:, 0].tolist()
        ry = scene["robot"].data.root_pos_w[:, 1].tolist()  # course axis (climb progress)
        rz = scene["robot"].data.root_pos_w[:, 2].tolist()
        contact_forces = scene["contact_forces"].data.net_forces_w
        per_body = torch.norm(contact_forces, dim=-1).amax(dim=0).tolist()
        episode_length = self.env.unwrapped.episode_length_buf.tolist()
        max_action = actions.abs().amax(dim=0).tolist()
        print(f"[PLAYDBG] step={step} root_x={[round(v,2) for v in rx]} "
              f"root_y={[round(v,2) for v in ry]} "
              f"root_z={[round(v,2) for v in rz]} dones={dones.sum().item()} "
              f"ep={episode_length} max|a|={round(max(max_action),3)}", flush=True)
        print(f"[PLAYDBG] max contact/body={[round(v,1) for v in per_body]}", flush=True)
        try:
            hits = scene["height_scanner"].data.ray_hits_w
            finite = torch.isfinite(hits).float().mean().item()
            print(f"[PLAYDBG] ray finite frac={finite:.3f} zmin={torch.nan_to_num(hits[...,2]).min().item():.2f} "
                  f"zmax={torch.nan_to_num(hits[...,2]).max().item():.2f}", flush=True)
        except Exception as err:  # noqa: BLE001
            print(f"[PLAYDBG] ray log failed: {err}", flush=True)
        tm = self.env.unwrapped.termination_manager
        print(f"[PLAYDBG] terms={tm.active_terms} dones={tm._term_dones.tolist()}", flush=True)
