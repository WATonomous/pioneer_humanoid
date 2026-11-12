"""Checks invoked by view_nasa_cargo.py --check; no learned policy or robot demonstration.

A velocity-controlled parallel-jaw fixture clamps the real handle colliders and lifts through
friction only. Placement is tested separately by dropping the bag above the tray.
"""
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from humanoid_isaac_scenes.nasa_cargo.scene import CARGO_POS, CARGO_ROT, TABLE_TOP_Z, TARGET_XY
from humanoid_isaac_scenes.nasa_cargo.task import placement_metrics


def configure_grasp_fixture(cfg):
    for name in ("test_finger_a", "test_finger_b"):
        setattr(cfg, name, RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + name,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.0, 0.0, 1.0)),
            spawn=sim_utils.CuboidCfg(
                size=(0.08, 0.012, 0.022),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.2),
            ),
        ))


def check(scene, sim, task):
    dt = sim.get_physics_dt()
    cargo = scene["cargo"]

    def step(n):
        result = None
        for _ in range(n):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)
            result = task.update(dt)
        return result

    result = step(200)
    assert torch.isfinite(cargo.data.root_state_w).all()
    assert torch.allclose(cargo.data.root_pos_w[0, :2], torch.tensor(CARGO_POS[:2], device=sim.device), atol=0.01)
    assert abs(result["bottom_height"][0].item() - TABLE_TOP_Z) < 0.01
    assert not result["success"][0], "pickup pose must not count as placement"
    start_height = cargo.data.root_pos_w[0, 2].item()
    start = cargo.data.root_pos_w[0].clone()

    def fingers(gap, lift, initialise=False):
        for name, sign in (("test_finger_a", -1), ("test_finger_b", 1)):
            finger = scene[name]
            target = start + start.new_tensor([0.0, 0.08002 + sign * gap, 0.303 + lift])
            if initialise:
                state = finger.data.default_root_state.clone()
                state[:, :3] = target
                finger.write_root_state_to_sim(state)
            else:
                velocity = torch.zeros((1, 6), device=sim.device)
                velocity[:, :3] = (60 * (target - finger.data.root_pos_w)).clamp(-0.5, 0.5)
                finger.write_root_velocity_to_sim(velocity)

    fingers(0.04, 0.0, initialise=True)
    for i in range(80):
        fingers(0.04 - 0.026 * i / 79, 0.0)
        step(1)
    for i in range(160):
        fingers(0.014, 0.15 * i / 159)
        step(1)
    rise = cargo.data.root_pos_w[0, 2].item() - start_height
    print(f"[Cargo check] handle grasp lift={rise:.4f} m; bag pose={cargo.data.root_state_w[0, :7].tolist()}", flush=True)
    assert rise > 0.08, f"contact grasp did not lift the bag: rise={rise:.4f} m"
    assert task.lifted[0], "lift event not detected"
    # Park fixture away. The placement test deliberately positions the bag above the
    # target; this is a geometry/settling check, not autonomous pick-and-place.
    for name in ("test_finger_a", "test_finger_b"):
        scene[name].write_root_state_to_sim(scene[name].data.default_root_state.clone())
    state = cargo.data.default_root_state.clone()
    state[:, :2] = state.new_tensor(TARGET_XY)
    state[:, 2] += 0.18
    state[:, 7:] = 0
    cargo.write_root_state_to_sim(state)
    cargo.reset()
    result = step(600)
    assert result["success"][0], f"settled target drop did not succeed: {result}"
    settled = cargo.data.root_state_w.clone()
    origins = scene.env_origins
    assert placement_metrics(settled, origins)["settled"][0]
    for label, altered in (("partial overhang", settled.clone()), ("hover", settled.clone()),
                           ("moving", settled.clone()), ("tipped", settled.clone())):
        if label == "partial overhang":
            altered[:, 0] += 0.15
        elif label == "hover":
            altered[:, 2] += 0.10
        elif label == "moving":
            altered[:, 7] = 0.5
        else:
            altered[:, 3:7] = altered.new_tensor([0.70710678, 0.70710678, 0, 0])
        assert not placement_metrics(altered, origins)["settled"][0], label
    task.reset()
    result = task.update(2.0)
    assert not result["success"][0], "task reset must clear prior lift and success"
    print(f"NASA_CARGO_CHECK_PASS handle_grasp_lift={rise:.4f} m; target drop, negative cases, reset passed", flush=True)
