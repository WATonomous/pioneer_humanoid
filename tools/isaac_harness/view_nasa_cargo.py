"""View NASA cargo pick-and-place with Pioneer arms; no VR, ROS or policy required.

    /workspace/isaaclab/isaaclab.sh -p tools/isaac_harness/view_nasa_cargo.py --device cuda
    # Physics/task check:
    ... --headless --check
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for package in ("src/simulation/isaac_scenes", "src/pioneer_humanoid"):
    sys.path.insert(0, str(ROOT / package))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=0, help="Exit after N steps (0: until window closes)")
parser.add_argument("--check", action="store_true", help="Run physical handle-grasp and placement checks, then exit")
parser.add_argument("--snapshot", type=Path, help="Save an initial view; requires --enable_cameras")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 0:
    parser.error("--steps must be nonnegative")
launcher = AppLauncher(args)
app = launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from humanoid_isaac_scenes.nasa_cargo.scene import CAMERA, ROBOT_POS, NasaCargoSceneCfg
from humanoid_isaac_scenes import make_scene_cfg
from humanoid_isaac_scenes.nasa_cargo.task import CargoTask
from pioneer_humanoid.bimanual_arm import BIMANUAL_ARM_CFG


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    sim.set_camera_view(*CAMERA)
    cfg = make_scene_cfg("nasa_cargo", BIMANUAL_ARM_CFG, env_spacing=3.0)
    if args.check:
        from check_nasa_cargo import configure_grasp_fixture
        configure_grasp_fixture(cfg)
    if args.snapshot:
        from isaaclab.sensors import CameraCfg
        cfg.preview = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Preview", width=960, height=720, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=20.0, horizontal_aperture=24.0),
        )
    scene = InteractiveScene(cfg)
    sim.reset()
    robot = scene["robot"]
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.set_joint_position_target(robot.data.default_joint_pos)
    scene.update(sim.get_physics_dt())
    task = CargoTask(scene)
    if args.check:
        from check_nasa_cargo import check
        check(scene, sim, task)
        return
    if args.snapshot:
        from PIL import Image
        camera = scene["preview"]
        camera.set_world_poses_from_view(
            torch.tensor([CAMERA[0]], device=sim.device), torch.tensor([CAMERA[1]], device=sim.device))
        for _ in range(8):
            sim.render()
            camera.update(sim.get_physics_dt(), force_recompute=True)
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(camera.data.output["rgb"][0, ..., :3].cpu().numpy()).save(args.snapshot)
    reset_requested = [False]
    if not args.headless:
        import carb
        import omni.appwindow

        def on_key(event, *unused):
            if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input.name == "R":
                reset_requested[0] = True
        keyboard_interface = carb.input.acquire_input_interface()
        keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        keyboard_subscription = keyboard_interface.subscribe_to_keyboard_events(keyboard, on_key)
    print("[Cargo viewer] Ready. Gray station: pickup. Green tray: target. R resets the bag.", flush=True)
    steps = 0
    while app.is_running() and (args.steps == 0 or steps < args.steps):
        started = time.monotonic()
        if reset_requested[0]:
            reset_requested[0] = False
            scene["cargo"].write_root_state_to_sim(scene["cargo"].data.default_root_state.clone())
            scene["cargo"].reset()
            task.reset()
        if sim.is_playing():
            scene.write_data_to_sim()
            sim.step(render=not args.headless)
            scene.update(sim.get_physics_dt())
            status = task.update(sim.get_physics_dt())
            steps += 1
            if steps % 400 == 0:
                phase = "success" if status["success"][0] else "lifted" if status["ever_lifted"][0] else "waiting for pickup"
                print(f"[Cargo task] {phase}", flush=True)
        else:
            sim.render()
        time.sleep(max(0.0, sim.get_physics_dt() - (time.monotonic() - started)))
    print(f"[Cargo viewer] Closed after {steps} steps.", flush=True)


try:
    main()
except Exception:
    traceback.print_exc()
    sys.stderr.flush()
    os._exit(1)
else:
    sys.stdout.flush()
    os._exit(0)
