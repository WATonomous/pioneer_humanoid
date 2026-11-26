"""Joint ranges, default pose, action scale/offset/clip and effort limits for the leg action term."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src/simulation/humanoid_rl/humanoid_rl/scripts"))
from isaaclab.app import AppLauncher
app = AppLauncher({"headless": True, "device": "cuda:0"}).app
import gymnasium as gym  # noqa: E402
import humanoid_rl_tasks  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
TASK = "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-Play-v0"
cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=2)
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg)); u = env.unwrapped
robot = u.scene["robot"]; at = u.action_manager.get_term("joint_pos")
names = list(at._joint_names)
jid = [robot.joint_names.index(n) for n in names]
lim = robot.data.joint_pos_limits[0].cpu()[jid]
dflt = robot.data.default_joint_pos[0].cpu()[jid]
eff = robot.data.joint_effort_limits[0].cpu()[jid] if hasattr(robot.data, "joint_effort_limits") else torch.zeros(len(jid))
def v(x):
    t = torch.as_tensor(x, dtype=torch.float32).cpu()
    return t[0] if t.dim() == 2 else (t.expand(len(names)) if t.dim() == 0 else t)
sc, off = v(at._scale), v(at._offset)
print("[J] name        lo      hi    default  offset  scale  effort | reach(|a|) to lo / hi")
for i, n in enumerate(names):
    lo, hi = float(lim[i, 0]), float(lim[i, 1])
    print(f"[J] {n:10s} {lo:+6.2f} {hi:+6.2f}  {float(dflt[i]):+6.2f}  {float(off[i]):+6.2f}  {float(sc[i]):5.2f} {float(eff[i]):6.0f} |"
          f" {(lo - float(off[i])) / float(sc[i]):+7.1f} / {(hi - float(off[i])) / float(sc[i]):+6.1f}")
kp = robot.data.joint_stiffness[0].cpu()[jid]
kd = robot.data.joint_damping[0].cpu()[jid]
print("[J] name        kp      kd    effort   error(rad) for full torque (effort/kp)")
for i, n in enumerate(names):
    print(f"[J] {n:10s} {float(kp[i]):7.1f} {float(kd[i]):6.1f} {float(eff[i]):6.0f}   {float(eff[i]) / max(float(kp[i]), 1e-6):6.2f}")
print("[J] clip:", getattr(at, "_clip", None) is not None)
print("[J] OK", flush=True)
env.close(); app.close()
