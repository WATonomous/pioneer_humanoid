"""Closed-loop RTC rollout of a flow-matching policy on the pioneer left arm, push-block scene.

Scene = the "push" entry of humanoid_isaac_scenes (same layout/assets as the RL push-block task),
plus the ego + wrist cameras from quest_cameras.py -- the exact cameras the push_box dataset was
recorded with (run_quest_bimanual_teleop.py). The scene is driven raw (set_joint_position_target),
not through the gym env, whose action/obs managers are RL-specific (scaled relative actions, no
cameras).

Timing matches the quest recording: 50 Hz physics, policy at 10 Hz (every 5th physics step).

NOT YET HOOKED UP TO THE VLA: no real checkpoint has been run through this. Before trusting a
real run, the VLA's schema, camera keys and action space (absolute vs delta) still have to be
matched -- see the TODO(VLA) block below.

    # simulation_isaac container
    /workspace/isaaclab/isaaclab.sh -p /workspace/humanoid/src/robot_learning/scripts/pioneer_push_eval_rtc.py \\
        --policy_path <checkpoint> --num_episodes 10
    # no checkpoint: hold the home pose, exercises scene/cameras/reset/scoring only
    ... pioneer_push_eval_rtc.py --dry_run --num_episodes 2 --episode_s 5 --save_frames /tmp/frames
"""

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher

_SRC = Path(__file__).resolve().parents[2]
# quest_cameras is not an installed package (run_quest_bimanual_teleop.py imports it the same way).
sys.path.insert(0, str(_SRC / "teleop" / "quest_teleop" / "sim"))
# Editable-installed in the image; fallbacks for a bare bind-mounted checkout.
sys.path.insert(0, str(_SRC / "pioneer_humanoid"))
sys.path.insert(0, str(_SRC / "robot_learning"))

parser = argparse.ArgumentParser(description="RTC rollout on the pioneer left arm, push-block scene.")
parser.add_argument("--policy_path", type=str, default=None, help="checkpoint folder or Hub repo id")
parser.add_argument(
    "--schema", type=str,
    default=str(_SRC / "robot_learning" / "config" / "dataset_schema_wato_arm_v2_push_box.yaml"),
    help="dataset schema the checkpoint was trained on",
)
parser.add_argument("--task_description", type=str, default="push the block into the box")
parser.add_argument("--execution_horizon", type=int, default=10)
parser.add_argument("--max_guidance_weight", type=float, default=1.0)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--episode_s", type=float, default=30.0, help="timeout per episode, seconds")
parser.add_argument("--settle_ticks", type=int, default=5, help="policy ticks to hold home after reset")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no_realtime", action="store_true", help="don't pace the loop to wall-clock time")
parser.add_argument("--dry_run", action="store_true", help="no policy: command the home pose")
parser.add_argument("--save_frames", type=str, default=None, help="dir to dump the first frame of each camera")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
if args_cli.policy_path is None and not args_cli.dry_run:
    parser.error("--policy_path is required unless --dry_run")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from humanoid_isaac_scenes import make_scene_cfg, scene_camera  # noqa: E402
from humanoid_rl_tasks.push_block.push_env_cfg import RewardsCfg, TerminationsCfg  # noqa: E402
from humanoid_robot_learning.pioneer_interface import (  # noqa: E402
    SIM_ACTION_JOINT_ORDER,
    PioneerLeftArmInterface,
)
from humanoid_robot_learning.rtc_driver import RTCDrivenPolicy  # noqa: E402
from humanoid_robot_learning.schema import enabled_images, load_yaml  # noqa: E402
from pioneer_humanoid.bimanual_arm import (  # noqa: E402
    BIMANUAL_ARM_CFG,
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_JOINTS,
    apply_joint_limits,
    resolve_joint_name,
)
from quest_cameras import make_ego_cam_cfg, make_wrist_cam_cfg  # noqa: E402

PHYSICS_DT = 0.02
DECIMATION = 5
POLICY_HZ = 1.0 / (PHYSICS_DT * DECIMATION)  # 10 Hz, the push_box recording fps

# dataset image key -> scene sensor name (run_quest_bimanual_teleop.py _capture_record_images)
CAMERA_SENSORS = {"ego": "ego_cam", "wrist": "wrist_cam"}
CAMERA_FACTORIES = {"ego_cam": make_ego_cam_cfg, "wrist_cam": make_wrist_cam_cfg}


def read_policy_state(robot, state_joint_ids: list[int]) -> torch.Tensor:
    """observation.state for this tick: (7,) absolute joint positions (rad) of env 0.

    Must match what the push_box dataset recorded (run_quest_bimanual_teleop.py ~L1634-1638):
    absolute (not joint_pos_rel), in the schema's joint_names order (joint1L..joint6l, joint7l;
    state_joint_ids is already in that order), no joint8l, read before stepping physics.
    A wrong order or joint set doesn't crash -- the policy just sees a scrambled arm.
    """
    return robot.data.joint_pos[0, state_joint_ids].clone()


# ======================================================================================
# TODO(VLA) -- hook up the VLA: confirm against the real checkpoint before trusting a real run:
#   - schema: joint list + gripper representation (push_box: joint7l position in metres;
#     pioneer_v1: one 0-1 closure value) -- pass the right --schema
#   - camera keys the checkpoint expects (push_box: ego/wrist; pioneer_v1: ego/wrist_left)
#   - absolute vs delta actions: the pi0.5 pipeline trains on deltas
#     (humanoid_il/pi0.5/action_space.py) -- predicted chunks would need to_absolute()
#     against the state each chunk was planned from, which RTCDrivenPolicy doesn't do yet
#   - dataset fps -> POLICY_HZ, and the task string
# ======================================================================================


def _joint_ids(robot, names: list[str]) -> list[int]:
    name_to_id = {name: i for i, name in enumerate(robot.data.joint_names)}
    return [name_to_id[resolve_joint_name(robot, name)] for name in names]


def build_scene(camera_keys: list[str]) -> tuple[sim_utils.SimulationContext, InteractiveScene]:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=PHYSICS_DT, render_interval=DECIMATION, device=args_cli.device)
    )
    sim.set_camera_view(*scene_camera("push"))

    scene_cfg = make_scene_cfg("push", BIMANUAL_ARM_CFG, num_envs=1, env_spacing=2.0)
    # Cameras after the robot: they're parented under its links, and entities are created in order.
    for key in camera_keys:
        sensor = CAMERA_SENSORS[key]
        setattr(scene_cfg, sensor, CAMERA_FACTORIES[sensor]())
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    apply_joint_limits(scene["robot"])
    return sim, scene


def read_visual_obs(scene: InteractiveScene, camera_keys: list[str]) -> dict[str, torch.Tensor]:
    # Batch dim kept: sim_obs_to_policy_processor indexes [0]. uint8 HWC, same as the recording.
    return {
        f"rgb_{key}": scene[CAMERA_SENSORS[key]].data.output["rgb"][..., :3].to(torch.uint8)
        for key in camera_keys
    }


def build_policy(interface: PioneerLeftArmInterface, initial_action: torch.Tensor, device: str) -> RTCDrivenPolicy:
    # Set before make_policy so the load-time RTC config and the driver's per-call
    # execution_horizon are the same number.
    os.environ["RTC_EXECUTION_HORIZON"] = str(args_cli.execution_horizon)
    os.environ["RTC_MAX_GUIDANCE_WEIGHT"] = str(args_cli.max_guidance_weight)
    interface.make_policy(args_cli.policy_path)

    rtc_config = getattr(interface.policy.config, "rtc_config", None)
    if rtc_config is None or not rtc_config.enabled:
        raise SystemExit(
            f"RTC did not get enabled on {type(interface.policy).__name__} -- not a flow-matching "
            "policy, or lerobot.policies.rtc failed to import. RTCDrivenPolicy would silently run unguided."
        )

    return RTCDrivenPolicy(
        policy=interface.policy,
        preprocessor=interface.preprocessor,
        postprocessor=interface.postprocessor,
        task_description=args_cli.task_description,
        robot_type=interface.robot_type,
        execution_horizon=args_cli.execution_horizon,
        control_hz=POLICY_HZ,
        initial_action=initial_action.unsqueeze(0),
        device=torch.device(device),
    )


def step_physics(sim, scene, robot, sim_action, sim_action_ids, right_ids, right_home) -> None:
    """Hold sim_action on the left arm and home on the right for one policy tick (DECIMATION steps)."""
    robot.set_joint_position_target(sim_action.unsqueeze(0), joint_ids=sim_action_ids)
    robot.set_joint_position_target(right_home, joint_ids=right_ids)
    for substep in range(DECIMATION):
        scene.write_data_to_sim()
        sim.step(render=False)
        # Render once per policy tick, before the update that reads the cameras
        # (same ordering as isaaclab's ManagerBasedEnv.step).
        if substep == DECIMATION - 1:
            sim.render()
        scene.update(PHYSICS_DT)


def reset_episode(scene, robot) -> None:
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.reset()

    block = scene["object"]
    root_state = block.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    block.write_root_state_to_sim(root_state)
    block.reset()


# Success / failure terms from the RL task cfg, so thresholds live in one place. These funcs only
# read env.scene (and env.scene.env_origins), so a namespace holding the raw scene stands in for env.
_SUCCESS_TERM = RewardsCfg().block_on_floor
_FAILURE_TERMS = {
    "off_course": TerminationsCfg().block_off_course,
    "dropped": TerminationsCfg().object_dropping,
}


def episode_outcome(env_shim) -> str | None:
    if bool(_SUCCESS_TERM.func(env_shim, **_SUCCESS_TERM.params)[0]):
        return "success"
    for name, term in _FAILURE_TERMS.items():
        if bool(term.func(env_shim, **term.params)[0]):
            return name
    return None


def save_frames(visual_obs: dict[str, torch.Tensor], out_dir: str) -> None:
    import imageio

    os.makedirs(out_dir, exist_ok=True)
    for key, img in visual_obs.items():
        path = os.path.join(out_dir, f"{key.removeprefix('rgb_')}.png")
        imageio.imwrite(path, img[0].cpu().numpy())
        print(f"[INFO]: saved {path}")


def main():
    torch.manual_seed(args_cli.seed)

    schema = load_yaml(Path(args_cli.schema))
    images = enabled_images(schema)
    unknown = sorted(set(images) - set(CAMERA_SENSORS))
    if unknown:
        raise SystemExit(f"{args_cli.schema}: no scene camera for images {unknown}; known: {list(CAMERA_SENSORS)}")
    camera_keys = list(images)

    sim, scene = build_scene(camera_keys)
    robot = scene["robot"]
    device = sim.device

    interface = PioneerLeftArmInterface(
        device=device,
        cameras={k: {"height": int(v["height"]), "width": int(v["width"])} for k, v in images.items()},
        schema_path=args_cli.schema,
        task_description=args_cli.task_description,
    )

    state_joint_ids = _joint_ids(robot, interface.joint_names)
    sim_action_ids = _joint_ids(robot, SIM_ACTION_JOINT_ORDER)
    right_ids = _joint_ids(robot, RIGHT_ARM_JOINTS + RIGHT_GRIPPER_JOINTS)
    right_home = robot.data.default_joint_pos[:, right_ids].clone()
    home_sim_action = robot.data.default_joint_pos[0, sim_action_ids].clone()
    initial_action = robot.data.default_joint_pos[0, state_joint_ids].clone()  # home, in policy space

    print(f"[INFO]: state joints  {[robot.data.joint_names[i] for i in state_joint_ids]}")
    print(f"[INFO]: action joints {[robot.data.joint_names[i] for i in sim_action_ids]}")
    print(f"[INFO]: cameras {camera_keys}, policy {POLICY_HZ:.0f} Hz, physics {1 / PHYSICS_DT:.0f} Hz")

    driver = None if args_cli.dry_run else build_policy(interface, initial_action, device)

    env_shim = SimpleNamespace(scene=scene)
    max_ticks = int(args_cli.episode_s * POLICY_HZ)
    tick_period = 1.0 / POLICY_HZ
    results = []
    frames_saved = False

    # no_grad, never inference_mode: RTC's guidance re-enables autograd inside predict_action_chunk.
    with torch.no_grad():
        for episode in range(args_cli.num_episodes):
            reset_episode(scene, robot)
            if driver is not None:
                driver.reset()
            for _ in range(args_cli.settle_ticks):
                step_physics(sim, scene, robot, home_sim_action, sim_action_ids, right_ids, right_home)

            outcome = "timeout"
            tick = -1
            next_tick = time.perf_counter()
            lag_warned = False
            for tick in range(max_ticks):
                if not simulation_app.is_running():
                    break

                state = read_policy_state(robot, state_joint_ids)
                visual_obs = read_visual_obs(scene, camera_keys)
                if args_cli.save_frames and not frames_saved:
                    save_frames(visual_obs, args_cli.save_frames)
                    frames_saved = True
                if tick == 0:
                    print(f"[INFO]: ep {episode + 1} start state {state.tolist()}")

                frame = interface.sim_obs_to_policy_processor(state, visual_obs)
                action = initial_action if driver is None else driver.get_action(frame)
                # The postprocessor may hand back CPU tensors; the interface clamps against device tensors.
                sim_action = interface.prediction_to_sim_processor(action.to(device), frame)

                step_physics(sim, scene, robot, sim_action, sim_action_ids, right_ids, right_home)

                result = episode_outcome(env_shim)
                if result is not None:
                    outcome = result
                    break

                # RTCDrivenPolicy converts wall-clock inference time to ticks via control_hz, which
                # is only right if ticks happen in real time -- so pace to it, and say when we can't.
                if not args_cli.no_realtime:
                    next_tick += tick_period
                    slack = next_tick - time.perf_counter()
                    if slack > 0:
                        time.sleep(slack)
                    else:
                        if not lag_warned and -slack > tick_period:
                            print(f"[WARN]: sim slower than real time ({-slack * 1000:.0f} ms behind) -- "
                                  "RTC delay estimates will be too large")
                            lag_warned = True
                        next_tick = time.perf_counter()

            results.append(outcome)
            successes = results.count("success")
            print(f"[INFO]: ep {episode + 1}/{args_cli.num_episodes}: {outcome} after {tick + 1} ticks "
                  f"-- success {successes}/{len(results)} ({100 * successes / len(results):.1f}%)")
            if not simulation_app.is_running():
                break

    successes = results.count("success")
    print(f"[INFO]: Evaluated {len(results)} episodes: {dict((r, results.count(r)) for r in set(results))}")
    print(f"[INFO]: Success Rate: {successes}/{len(results)} ({100 * successes / max(len(results), 1):.1f}%)")


if __name__ == "__main__":
    main()
    simulation_app.close()
