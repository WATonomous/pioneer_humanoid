# Connect the existing MuJoCo scenes and Quest driver to HG-DAgger.
from pathlib import Path
import sys
import time

import numpy as np

from ..core import Contract, Signals


# Make the existing robot, scene registry and Quest driver importable.
def source_paths():
    root = Path(__file__).resolve().parents[5]
    for relative in ("src/pioneer_humanoid", "src/simulation/mujoco_scenes",
                     "src/teleop/quest_controller_teleop"):
        path = root / relative
        if not path.is_dir():
            raise FileNotFoundError(f"MuJoCo/Quest adapter requires a source checkout containing {path}")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


# Adapt an existing MuJoCo scene to the collection and evaluation interface.
class MuJoCoScene:
    # Load the requested scene and prepare its joint, camera and timing settings.
    def __init__(self, scene, arms=("left", "right"), fps=25, cameras=None, headless=False):
        source_paths()
        import mujoco
        import humanoid_mujoco_scenes as registry
        from pioneer_humanoid import arm_8dof_params as params

        if not arms or len(set(arms)) != len(arms) or any(s not in ("left", "right") for s in arms):
            raise ValueError("arms must be a nonempty ordered subset of left/right")
        self.mj, self.registry, self.scene_name, self.arms = mujoco, registry, scene, tuple(arms)
        self.headless = headless
        self.sizes = {k: tuple(v) for k, v in (cameras if cameras is not None else {
            "ego": [240, 320], **{f"wrist_{s}": [240, 320] for s in arms}}).items()}
        self.model = registry.make_model(scene, cameras=self.sizes, robot="8dof")
        self.data = mujoco.MjData(self.model)
        self.substeps = round(1 / fps / self.model.opt.timestep)
        if self.substeps < 1 or not np.isclose(self.substeps * self.model.opt.timestep, 1 / fps):
            raise ValueError("fps must divide the physics clock exactly")
        self.names = {"left": params.LEFT_ARM_JOINTS, "right": params.RIGHT_ARM_JOINTS}
        self.qpos = {s: [int(self.model.joint(n).qposadr[0]) for n in self.names[s]] for s in self.names}
        self.dofs = {s: [int(self.model.joint(n).dofadr[0]) for n in self.names[s]] for s in self.names}
        self.acts = {s: [self.model.actuator(n).id for n in self.names[s]] for s in self.names}
        self.grips = {s: self.model.actuator(f"{s}_gripper").id for s in self.names}
        names = tuple(n for s in arms for n in [*self.names[s], f"{s}_gripper"])
        units = tuple(u for _ in arms for u in [*["rad"] * 7, "closure_0_1"])
        self.contract = Contract(scene, "pioneer_bimanual_arm_8dof", fps, names, names, units, units,
                                 {f"observation.images.{k}": (*size, 3) for k, size in self.sizes.items()},
                                 "absolute_joint_radians_and_gripper_0_1")
        limits = params.urdf_limits()
        self.limits = np.array([limits[n] for n in names], dtype=np.float32)
        self.limits[7::8] = [0, 1]
        self.speed = np.array([v for _ in arms for v in [.9, .9, .9, .9, 3.6, 3.6, 3.6, .2 / .041]])
        self.renderers = {}
        self.previous = {}
        self.reset(0)

    # Read the measured arm joint positions.
    def joint_positions(self):
        return {s: self.data.qpos[self.qpos[s]].copy() for s in self.names}

    # Reset the robot and scene randomization with the supplied seed.
    def reset(self, seed):
        from pioneer_humanoid.mujoco_bimanual_arm_8dof import set_home
        self.mj.mj_resetData(self.model, self.data)
        set_home(self.model, self.data)
        hook = self.registry.scene_reset(self.scene_name)
        if hook:
            hook(self.model, self.data, np.random.default_rng(seed))
        self.mj.mj_forward(self.model, self.data)
        self.previous = self.joint_positions()
        self.command = np.array([self.data.qpos[int(self.model.joint(n).qposadr[0])]
                                 for n in self.contract.action_names], dtype=np.float32)
        self.command[7::8] = np.clip(self.command[7::8] / .041, 0, 1)
        # Keep any unused arm at its home pose too.
        for s in self.names:
            self.data.ctrl[self.acts[s]] = self.previous[s]
            self.data.ctrl[self.grips[s]] = 0

    # Read arm state, normalized gripper closure and RGB camera images.
    def observe(self):
        obs = {"observation.state": np.array([
            self.data.qpos[int(self.model.joint(n).qposadr[0])] for n in self.contract.state_names], dtype=np.float32)}
        obs["observation.state"][7::8] = np.clip(obs["observation.state"][7::8] / .041, 0, 1)
        for name, (height, width) in self.sizes.items():
            if name not in self.renderers:
                self.renderers[name] = self.mj.Renderer(self.model, height=height, width=width)
            renderer = self.renderers[name]
            renderer.update_scene(self.data, camera=name)
            obs[f"observation.images.{name}"] = renderer.render().copy()
        return obs

    # Limit the targets, advance physics and return the executed command.
    def execute(self, action):
        action = self.contract.action(action)
        dt = 1 / self.contract.fps
        action = np.clip(action, self.limits[:, 0], self.limits[:, 1])
        action = np.clip(action, self.command - self.speed * dt, self.command + self.speed * dt).astype(np.float32)
        targets = {s: self.previous[s].copy() for s in self.names}
        velocities = {s: np.zeros(7) for s in self.names}
        for i, s in enumerate(self.arms):
            q = action[8*i:8*i+7]
            targets[s] = q
            velocities[s] = (q - self.previous[s]) / dt
            self.data.ctrl[self.grips[s]] = action[8*i+7] * .041
        for _ in range(self.substeps):
            # Use the upstream teleop's gravity and velocity compensation.
            for s, q in targets.items():
                acts = self.acts[s]
                kp = self.model.actuator_gainprm[acts, 0]
                kv = -self.model.actuator_biasprm[acts, 2]
                self.data.ctrl[acts] = q + (kv * velocities[s] + self.data.qfrc_bias[self.dofs[s]]) / kp
            hook = self.registry.scene_step(self.scene_name)
            if hook:
                hook(self.model, self.data)
            self.mj.mj_step(self.model, self.data)
        self.previous = {s: q.copy() for s, q in targets.items()}
        self.command = action.copy()
        return action

    # Use the selected scene's success rule for evaluation.
    def success(self):
        hook = self.registry.scene_success(self.scene_name)
        if hook is None:
            raise ValueError(f"Scene {self.scene_name!r} has no success predicate; add @scene(success=...)")
        return bool(hook(self.model, self.data))

    # Show camera views and report whether the operator closed the window.
    def preview(self, status):
        if self.headless or not self.sizes:
            return False
        import cv2
        obs = self.observe()
        images = [obs[f"observation.images.{k}"] for k in self.sizes]
        width = max(x.shape[1] for x in images)
        image = np.vstack([cv2.resize(x, (width, round(x.shape[0] * width / x.shape[1]))) for x in images])
        image = image[:, :, ::-1].copy()
        cv2.putText(image, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
        cv2.imshow("HG-DAgger", image)
        return (cv2.waitKey(1) & 0xff) == 27 or cv2.getWindowProperty("HG-DAgger", cv2.WND_PROP_VISIBLE) < 1

    # Close the camera renderers and preview windows.
    def close(self):
        for renderer in self.renderers.values():
            renderer.close()
        if not self.headless:
            import cv2
            cv2.destroyAllWindows()


# Map the existing Quest driver to human actions and collection decisions.
class QuestTeleop:
    # Configure and start the upstream Quest controller driver.
    def __init__(self, scene, controller=None):
        import argparse
        from scene_driver import ControllerSceneDriver, add_controller_args, check_controller_args

        self.scene = scene
        parser = argparse.ArgumentParser()
        add_controller_args(parser)
        parser.set_defaults(controller_frame="base")
        args = parser.parse_args([])
        for k, v in (controller or {}).items():
            if not hasattr(args, k):
                raise ValueError(f"Unknown upstream controller option {k}")
            setattr(args, k, v)
        check_controller_args(parser, args)
        calibration = None
        if args.controller_frame == "camera":
            base = scene.data.xmat[scene.model.body("base_link").id].reshape(3, 3)
            camera = scene.data.cam_xmat[scene.model.camera("ego").id].reshape(3, 3)
            calibration = base.T @ camera
        self.driver = ControllerSceneDriver.from_args(args, {s: scene.names[s] for s in scene.arms}, calibration)
        self.driver.start_thread()
        self.presses = 0
        print(f"[HG-DAgger VR] adb reverse tcp:{args.controller_port} tcp:{args.controller_port}; "
              f"Quest: http://localhost:{args.controller_port}/ -> Enter VR", flush=True)
        print("[HG-DAgger VR] Right grip: intervene; release: policy; "
              "right B: save/reset; left Y: discard/reset", flush=True)

    # Reanchor the controllers to the measured arm positions.
    def seed(self):
        self.driver.reset(self.scene.joint_positions())

    # Read tracking, grip takeover and latched save or discard presses.
    def poll(self):
        discard = self.driver.update(self.scene.joint_positions())
        with self.driver.shared.lock:
            shared = self.driver.shared
            packet, received, presses = shared.packet, shared.received, shared.record_presses
        accept = presses != self.presses
        self.presses = presses
        ready = (packet is not None and time.monotonic() - received <= .2 and packet["head"] is not None
                 and all(packet[s] is not None for s in {"right", *self.scene.arms}))
        takeover = bool(ready and packet["right"].clutch)
        return Signals(bool(ready), takeover, accept, discard)

    # Convert controller targets and triggers into arm and gripper commands.
    def action(self):
        targets = self.driver.solve(self.scene.joint_positions(), 1 / self.scene.contract.fps)
        return np.array([x for s in self.scene.arms for x in [
            *targets[s], self.driver.grippers[s]]], dtype=np.float32)

    # Show collection status through the existing Quest feedback channel.
    def feedback(self, status):
        with self.driver.shared.lock:
            self.driver.shared.feedback["status"] = status

    # Stop the Quest controller service.
    def close(self):
        self.driver.close()


# Create a scene adapter using the application settings.
def make_scene(config):
    return MuJoCoScene(**config)


# Create a Quest adapter for the selected scene.
def make_teleop(scene, config):
    return QuestTeleop(scene, **config)
