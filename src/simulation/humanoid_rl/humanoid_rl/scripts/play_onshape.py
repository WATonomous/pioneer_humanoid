# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False,
                    help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200,
                    help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None,
                    help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False,
                    help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import importlib.metadata as metadata
import math
import os
import time
import torch

from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import humanoid_rl_tasks  # noqa: F401  (registers every RL task)
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

installed_rsl_rl_version = metadata.version("rsl-rl-lib")


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
    constant on the stairs, so a camera that simply copies the root's position
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


def main():
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # Symmetry experiments: override the *spawn* joint pose without touching the
    # shipped config, e.g. PLAY_INIT_OVERRIDE="Ankle_P_R=-0.10,Hip_A_R=-0.04".
    _init_ovr = os.environ.get("PLAY_INIT_OVERRIDE", "").strip()
    if _init_ovr:
        _jp = env_cfg.scene.robot.init_state.joint_pos
        for _kv in _init_ovr.split(","):
            if not _kv.strip():
                continue
            _k, _v = _kv.split("=")
            _k = _k.strip()
            _old = _jp.get(_k)
            _jp[_k] = float(_v)
            print(f"[PLAYGEO] init override {_k}: {_old} -> {_jp[_k]}", flush=True)

    # specify directory for logging experiments
    _out_root = os.environ.get("HUMANOID_ROOT") or os.getcwd()
    log_root_path = os.path.abspath(os.path.join(_out_root, "outputs", "rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # Where a recording lands. PLAY_VIDEO_DIR / PLAY_VIDEO_PREFIX let a second
    # angle be recorded next to the first instead of overwriting it. (Defined
    # before the recorder below, which uses it.)
    _video_dir = os.environ.get("PLAY_VIDEO_DIR") or os.path.join(log_dir, "videos", "play")
    _video_prefix = os.environ.get("PLAY_VIDEO_PREFIX", "rl-video")
    print(f"[PLAY] video -> {_video_dir} (prefix {_video_prefix})", flush=True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Lead-foot force test: step the spawn along the approach so each env meets
    # the first riser at a *different* gait phase. reset_root_state_uniform adds
    # env.scene.env_origins to the sampled spawn on every reset, so shifting the
    # per-env origin y is a persistent spawn offset with no change to the reset
    # event (and no change to the training recipe). PLAY_SPAWN_DY is a
    # comma-separated list, one value per env (repeats if fewer than num_envs).
    _dy_spec = os.environ.get("PLAY_SPAWN_DY", "").strip()
    if _dy_spec:
        _dys = [float(v) for v in _dy_spec.split(",") if v.strip()]
        _origins = env.unwrapped.scene.env_origins
        for _i in range(_origins.shape[0]):
            _origins[_i, 1] += _dys[_i % len(_dys)]
        print(f"[PLAYGEO] PLAY_SPAWN_DY={_dy_spec} -> env_origins="
              f"{[[round(v, 2) for v in row.tolist()] for row in _origins]}", flush=True)

    # pin the camera on the action (Onshape course spans +Y; robots start
    # staggered across the 10 m width near the marker). Also applies under
    # --video: rgb_array frames are rendered from this same viewport camera, so
    # skipping the pin here is what would put the recorder on a stray view.
    try:
        env.unwrapped.sim.set_camera_view(
            eye=(2.0, -6.0, 2.8), target=(4.4, 0.5, 0.8))
        from isaacsim.core.utils.viewports import set_active_viewport_camera
        set_active_viewport_camera("/OmniverseKit_Persp")
        print("[PLAY] camera pinned on Onshape course", flush=True)
    except Exception as err:  # noqa: BLE001 - headless has no viewport
        print(f"[PLAY] camera pin skipped: {err}", flush=True)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": _video_dir,
            "name_prefix": _video_prefix,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # A visible marker at the spawn point, so a reviewer can see *where* the
    # robot is supposed to start. The Onshape marker sliver itself is excluded
    # from the merged terrain (it is a spawn pad without collision), so it is
    # invisible in the render and there is nothing to check the spawn against.
    _marker_pos = os.environ.get("PLAY_MARKER", "").strip()
    if _marker_pos:
        try:
            _mx, _my, _mz = (float(v) for v in _marker_pos.split(","))
            from pxr import Gf, UsdGeom

            _stage = env.unwrapped.sim.stage
            _sphere = UsdGeom.Sphere.Define(_stage, "/World/spawn_marker")
            _sphere.CreateRadiusAttr(0.09)
            UsdGeom.Xformable(_sphere).AddTranslateOp().Set(Gf.Vec3d(_mx, _my, _mz + 0.06))
            print(f"[PLAYMARK] marker ball at ({_mx:.3f}, {_my:.3f}, {_mz:.3f})", flush=True)
        except Exception as err:  # noqa: BLE001 - diagnostic only
            print(f"[PLAYMARK] failed: {err}", flush=True)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if version.parse(installed_rsl_rl_version) >= version.parse("4.0.0"):
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        policy_nn = None
    else:
        policy_nn = runner.alg.policy if version.parse(installed_rsl_rl_version) >= version.parse("2.3.0") else runner.alg.actor_critic
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    dbg_step = 0
    # Opt-in follow-cam (PLAY_FOLLOW_ROBOT=1): keep robot 0 in frame so the gait
    # is visible as it climbs away from the fixed spawn view.
    _follow_cam = os.environ.get("PLAY_FOLLOW_ROBOT", "") not in ("", "0")
    # Rear cam (PLAY_REAR_CAM=1): directly behind robot 0 looking up the course,
    # so world x (the lateral axis here) is screen-horizontal and a sideways
    # foot swing is visible rather than foreshortened as it is from the side.
    _rear_cam = os.environ.get("PLAY_REAR_CAM", "") not in ("", "0")
    _follow_cam = _follow_cam or _rear_cam
    # Follow-cam geometry and smoothing (see _follow_cam_step).
    _cam_mode = os.environ.get("PLAY_CAM_MODE", "rear" if _rear_cam else "side").strip().lower()
    _cam_alpha = float(os.environ.get("PLAY_CAM_ALPHA", "0.05") or 0.05)
    _cam_dist = float(os.environ.get("PLAY_CAM_DIST", "3.4") or 3.4)
    _cam_height = float(os.environ.get("PLAY_CAM_HEIGHT", "0.55") or 0.55)
    _cam_look = float(os.environ.get("PLAY_CAM_LOOK", "1.2") or 1.2)
    _cam = {"eye": None, "target": None}
    # Shove detector (on by default when the push event is enabled).
    _push_detect = os.environ.get("PLAY_PUSH_DETECT", "1") not in ("", "0")
    _push_state = {"v": None}
    # Hard step cap for headless probes (PLAY_MAX_STEPS=N): exits cleanly without
    # the --video recorder (which is otherwise the only way to bound the loop).
    _max_steps = int(os.environ.get("PLAY_MAX_STEPS", "0") or 0)
    # Per-stride, per-foot swing stats for robot 0 (see _swing_arc_step).
    _swing = {"legs": {}, "err": None}
    # Per-env first-step-up (lead-foot) tracker (see _lead_step).
    _lead = {}
    print(f"[PLAYARC] joint_names={list(env.unwrapped.scene['robot'].data.joint_names)}", flush=True)
    print(f"[PLAYARC] body_names={list(env.unwrapped.scene['robot'].data.body_names)}", flush=True)
    # One-shot spawn geometry: the init pose is symmetric, so both feet should
    # sit symmetric about the root. This is the frame reference that tells us how
    # much of a walking lateral offset is real splay vs a body/root frame offset.
    _geo_done = False
    # Per-episode reward decomposition, printed at each reset: shows which terms
    # actually drive this policy on this course (the input to reward tuning).
    _ep_names = None
    _ep_sum = None
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            dbg_step += 1
            if not _geo_done and dbg_step >= 2:
                _geo_done = True
                try:
                    rob = env.unwrapped.scene["robot"]
                    bn = list(rob.data.body_names)
                    rt = rob.data.root_pos_w[0]
                    fl = (rob.data.body_pos_w[0, bn.index("Foot_L")] - rt).tolist()
                    fr = (rob.data.body_pos_w[0, bn.index("Foot_R")] - rt).tolist()
                    print(f"[PLAYGEO] spawn foot_L=({fl[0]:+.3f},{fl[1]:+.3f},{fl[2]:+.3f}) "
                          f"foot_R=({fr[0]:+.3f},{fr[1]:+.3f},{fr[2]:+.3f}) "
                          f"root=({float(rt[0]):+.3f},{float(rt[1]):+.3f},{float(rt[2]):+.3f}) "
                          f"heading={float(rob.data.heading_w[0]):+.3f} "
                          f"pg=({float(rob.data.projected_gravity_b[0,0]):+.3f},"
                          f"{float(rob.data.projected_gravity_b[0,1]):+.3f},"
                          f"{float(rob.data.projected_gravity_b[0,2]):+.3f})", flush=True)
                except Exception as err:  # noqa: BLE001 - diagnostic only
                    print(f"[PLAYGEO] failed: {err}", flush=True)
                # Direct symmetry audit at spawn, in world coordinates (no link
                # frame conventions involved): for each left/right body pair,
                # report the lateral midpoint offset from the root -- zero for a
                # mirror-symmetric robot at a symmetric pose -- and the
                # orientation residual against the mirrored right body
                # (quaternion mirror about the world x=const plane is (w,-x,y,z)).
                try:
                    rb = env.unwrapped.scene["robot"]
                    bn = list(rb.data.body_names)
                    rt = rb.data.root_pos_w[0]
                    print("[PLAYMIRROR] pair    mid_x-root      dpos_y   dpos_z    d_quat", flush=True)
                    for nm in ("Hip_F", "Hip_A", "Hip_R", "Calf", "Ankle", "Foot"):
                        il = bn.index(nm + "_L") if nm + "_L" in bn else None
                        ir = bn.index(nm + "_R") if nm + "_R" in bn else None
                        if il is None or ir is None:
                            continue
                        pl = rb.data.body_pos_w[0, il]
                        pr = rb.data.body_pos_w[0, ir]
                        mid = 0.5 * float(pl[0] + pr[0]) - float(rt[0])
                        dy = float(pl[1] - pr[1])
                        dz = float(pl[2] - pr[2])
                        ql = rb.data.body_quat_w[0, il]
                        qr = rb.data.body_quat_w[0, ir]
                        qm = [qr[0], -qr[1], qr[2], qr[3]]
                        dq = min(max(abs(float(ql[k]) - float(qm[k])) for k in range(4)),
                                 max(abs(float(ql[k]) + float(qm[k])) for k in range(4)))
                        print(f"[PLAYMIRROR] {nm:7} {mid:+8.4f}  {dy:+8.4f} {dz:+8.4f}  {dq:8.4f}", flush=True)
                    print(f"[PLAYMIRROR] root=({float(rt[0]):+.4f},{float(rt[1]):+.4f},{float(rt[2]):+.4f}) "
                          f"-> a symmetric spawn needs mid_x-root = 0 and d_quat ~ 0 for every pair",
                          flush=True)
                except Exception as err:  # noqa: BLE001 - diagnostic only
                    print(f"[PLAYMIRROR] failed: {err}", flush=True)
            # Reward decomposition for robot 0, accumulated across the episode.
            rm = env.unwrapped.reward_manager
            if _ep_names is None:
                _ep_names = list(getattr(rm, "_term_names",
                                         [f"term_{i}" for i in range(rm._step_reward.shape[1])]))
                _ep_sum = torch.zeros(len(_ep_names), dtype=torch.float64, device=rm._step_reward.device)
            _ep_sum += rm._step_reward[0].to(torch.float64)
            # Catch the exact reset: the 200-step debug schedule below lands
            # after the term flags have cleared, so a falling policy just shows
            # up as a shrinking episode counter with no cause.
            if dones.any():
                tm = env.unwrapped.termination_manager
                ry = env.unwrapped.scene["robot"].data.root_pos_w[:, 1].tolist()
                rz = env.unwrapped.scene["robot"].data.root_pos_w[:, 2].tolist()
                rx = env.unwrapped.scene["robot"].data.root_pos_w[:, 0].tolist()
                # On a spiral the polar coordinates are the meaningful ones:
                # the radius says how far off the walking line the robot is,
                # the angle says where it is around the staircase.
                polar = [f"r={math.hypot(x, y):.2f}@{math.degrees(math.atan2(y, x)):.0f}deg"
                         for x, y in zip(rx, ry)]
                print(f"[PLAYRESET] step={dbg_step} root_y={[round(v, 2) for v in ry]} "
                      f"root_z={[round(v, 2) for v in rz]} polar={polar} "
                      f"terms={tm.active_terms} "
                      f"dones={tm._term_dones.tolist()}", flush=True)
                ranked = torch.argsort(_ep_sum.abs(), descending=True)
                print("[PLAYREW] " + " ".join(
                    f"{_ep_names[i]}={_ep_sum[i].item():+.1f}" for i in ranked[:14]), flush=True)
                # The reward manager stores the *weighted* step reward, so a
                # sign question about a term needs its raw value. Call the
                # path-relative terms directly.
                try:
                    raw = []
                    for _i, _name in enumerate(rm._term_names):
                        if not _name.startswith("spiral"):
                            continue
                        _cfg = rm._term_cfgs[_i]
                        _value = _cfg.func(env.unwrapped, **_cfg.params)
                        _weighted = rm._step_reward[0, _i].item()
                        raw.append(f"{_name}: raw={_value[0].item():+.4f} weighted={_weighted:+.5f}")
                    if raw:
                        print("[PLAYRAW] " + " | ".join(raw), flush=True)
                except Exception as err:  # noqa: BLE001 - diagnostic only
                    print(f"[PLAYRAW] failed: {err}", flush=True)
                _ep_sum.zero_()
                # Record each finished env's lead-foot result before clearing it.
                _le = _lead.get("envs", {})
                for i in torch.nonzero(dones).flatten().tolist():
                    st = _le.get(i)
                    if st is not None:
                        print(f"[PLAYLEAD] env={i} episode_end lead={st['lead']} "
                              f"shuffle_steps={st['steps']} seq={''.join(st['seq'])} "
                              f"max_foot_z={st['max_z']:+.3f}", flush=True)
                        _le.pop(i, None)
            if _follow_cam:
                _follow_cam_step(
                    env, _cam, mode=_cam_mode, alpha=_cam_alpha,
                    dist=_cam_dist, height=_cam_height, look=_cam_look,
                )
            # Shove detector: the push event sets the root velocity in a single
            # step, so a jump in the root's speed is the shove. Printed so a
            # clip can be checked for "did the disturbance actually fire" and
            # where in the clip it happened.
            if _push_detect:
                vv = env.unwrapped.scene["robot"].data.root_lin_vel_w[0].tolist()
                prev = _push_state.get("v")
                if prev is not None and not bool(dones.any()):
                    dvx, dvy = vv[0] - prev[0], vv[1] - prev[1]
                    if max(abs(dvx), abs(dvy)) > 0.25:
                        py = float(env.unwrapped.scene["robot"].data.root_pos_w[0][1])
                        print(f"[PLAYPUSH] step={dbg_step} t={dbg_step * dt:.2f}s "
                              f"dv=({dvx:+.3f},{dvy:+.3f}) v=({vv[0]:+.2f},{vv[1]:+.2f}) "
                              f"root_y={py:+.2f}", flush=True)
                _push_state["v"] = vv
            if not _swing["err"]:
                _swing_arc_step(env, _swing)
            if not _lead.get("err"):
                _lead_step(env, _lead)
            if dbg_step % 200 == 0:
                sc = env.unwrapped.scene
                rx = sc["robot"].data.root_pos_w[:, 0].tolist()
                ry = sc["robot"].data.root_pos_w[:, 1].tolist()  # course axis (climb progress)
                rz = sc["robot"].data.root_pos_w[:, 2].tolist()
                cf = sc["contact_forces"].data.net_forces_w
                per_body = torch.norm(cf, dim=-1).amax(dim=0).tolist()
                ed = env.unwrapped.episode_length_buf.tolist()
                am = actions.abs().amax(dim=0).tolist()
                print(f"[PLAYDBG] step={dbg_step} root_x={[round(v,2) for v in rx]} "
                      f"root_y={[round(v,2) for v in ry]} "
                      f"root_z={[round(v,2) for v in rz]} dones={dones.sum().item()} "
                      f"ep={ed} max|a|={round(max(am),3)}", flush=True)
                print(f"[PLAYDBG] max contact/body={[round(v,1) for v in per_body]}", flush=True)
                try:
                    hits = sc["height_scanner"].data.ray_hits_w
                    ff = torch.isfinite(hits).float().mean().item()
                    print(f"[PLAYDBG] ray finite frac={ff:.3f} zmin={torch.nan_to_num(hits[...,2]).min().item():.2f} "
                          f"zmax={torch.nan_to_num(hits[...,2]).max().item():.2f}", flush=True)
                except Exception as err:  # noqa: BLE001
                    print(f"[PLAYDBG] ray log failed: {err}", flush=True)
                tm = env.unwrapped.termination_manager
                print(f"[PLAYDBG] terms={tm.active_terms} dones={tm._term_dones.tolist()}", flush=True)
            if version.parse(installed_rsl_rl_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            elif policy_nn is not None:
                policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        if _max_steps and dbg_step >= _max_steps:
            print(f"[PLAY] step cap reached ({_max_steps}) -- exiting", flush=True)
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

# From Isaac Lab scripts/reinforcement_learning/rsl_rl/play.py
