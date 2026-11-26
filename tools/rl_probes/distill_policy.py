"""Distill a saturated policy into a re-parameterised (well-conditioned) action space.

The teacher was trained with per-joint action scales that put the joint's window
at |a| of 2-17 while it sat at |a| ~ 8-10, i.e. past the window edge where the
reward is flat in the action. The student acts in the new space (SPIRAL_ACT_RANGE:
|a| = R reaches the far edge of every window, targets clipped to the window).

Teacher's effective command  = clip(offset + old_scale * a_teacher, lo, hi)
Student's regression target  = (that - offset) / new_scale       (the same angle)

The student starts as the teacher with its last layer rescaled by old_scale /
new_scale per output (exactly equivalent *before* clipping), then is trained
DAgger-style: the student drives the env (so it sees its own state distribution)
and the teacher labels every observation. The loss is in the new action units, on
the clipped target, so any output the teacher had pushed past the window is pulled
back inside it -- which is the point.

Env vars: TEACHER (checkpoint), OUT (output .pt), DISTILL_N, DISTILL_STEPS,
SPIRAL_ACT_RANGE must be set (e.g. 3).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "src/simulation/humanoid_rl/humanoid_rl/scripts"))

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True, "device": "cuda:0"})
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import humanoid_rl_tasks  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import cli_args  # noqa: E402

RAW = bool(os.environ.get("RAW"))   # control: teacher drives its own native action space, unmodified
assert RAW or float(os.environ.get("SPIRAL_ACT_RANGE", "0")) > 0, "set SPIRAL_ACT_RANGE (e.g. 3)"
TASK = os.environ.get("TASK_OVERRIDE", "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-v0")
TEACHER = os.environ.get("TEACHER", "/workspace/humanoid/outputs/rl/pioneer_humanoid_spiral_foothold/2026-11-21_20-09-43/model_399.pt")
OUT = os.environ.get("OUT", "/workspace/humanoid/outputs/rl/distilled/model_distilled.pt")
N = int(os.environ.get("DISTILL_N", "1000"))
STEPS = int(os.environ.get("DISTILL_STEPS", "6000"))
LR = float(os.environ.get("DISTILL_LR", "5e-4"))

# the teacher's action scales (SPIRAL_KNEE_SCALE=0.05 lineage) and the window each joint is clipped to
OLD_SCALE = {"Hip_F": 0.25, "Hip_A": 0.06, "Hip_R": 0.15, "Knee": 0.05, "Ankle_P": 0.12, "Ankle_R": 0.08}
# joint range +- effort/kp (kp 200/150/40, effort 222/120/60): the full torque range stays reachable at any
# joint angle. The knee keeps the teacher's own clip. Clipping to the bare joint range instead cannot climb
# (TEACHER_ONLY: 0 % completion), because a target past the limit is how the policy commands torque.
WINDOW = {"Hip_F": (-1.46, 2.68, 0.25), "Hip_A": (-0.97, 1.85, 0.0), "Hip_R": (-0.92, 1.97, 0.0),
          "Knee": (-1.10, -0.05, -0.5), "Ankle_P": (-2.02, 2.37, -0.25), "Ankle_R": (-1.76, 1.76, 0.0)}
if os.environ.get("SPIRAL_WINDOW_BARE"):
    WINDOW = {'Hip_F': (-0.35, 1.57, 0.25), 'Hip_A': (-0.17, 1.05, 0.0), 'Hip_R': (-0.52, 1.57, 0.0), 'Knee': (-1.1, -0.05, -0.5), 'Ankle_P': (-0.52, 0.87, -0.25), 'Ankle_R': (-0.26, 0.26, 0.0)}
_WIDE = float(os.environ.get("SPIRAL_WINDOW_WIDE", "0"))
if _WIDE:
    WINDOW = {k: ((-_WIDE, _WIDE, v[2]) if k != "Knee" else v) for k, v in WINDOW.items()}

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
cfg.scene.env_spacing = 0.0
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg))
u = env.unwrapped
at = u.action_manager.get_term("joint_pos")
names = list(at._joint_names)
dev = u.device
key = [next(k for k in OLD_SCALE if n.startswith(k)) for n in names]
old_sc = torch.tensor([OLD_SCALE[k] for k in key], device=dev)
lo = torch.tensor([WINDOW[k][0] for k in key], device=dev)
hi = torch.tensor([WINDOW[k][1] for k in key], device=dev)
off = torch.tensor([WINDOW[k][2] for k in key], device=dev)
new_sc = torch.as_tensor(at._scale, device=dev, dtype=torch.float32)
new_sc = new_sc[0] if new_sc.dim() == 2 else new_sc
print("[D] joints:", names, flush=True)
print("[D] old scale:", [round(float(x), 3) for x in old_sc], flush=True)
print("[D] new scale:", [round(float(x), 3) for x in new_sc], flush=True)


# The policy observes its previous action (obs slice [36:48], raw units). Changing the action scale
# rescales that input, so the teacher must be shown the last action in *its* units
# (a_old = a_new * new_scale / old_scale) or it sees a signal 3x smaller than it was trained on and
# falls -- which is what made both the teacher-only test and the first distillation fail.
LA = slice(36, 48)
ratio = (new_sc / old_sc).to(dev)


PREV_A = None   # the teacher's own previous (unclipped) output, in its own units


def teacher_obs(o):
    """Observation for the teacher, with its last-action input = its own previous raw output.

    Reconstructing it from the env's stored action (rescaled) is wrong: that action is clipped, and
    this policy feeds large outputs (|a| 8-22) back as its own memory.
    """
    o2 = o.clone()
    if PREV_A is None:
        o2["policy"][:, LA] = o2["policy"][:, LA] * ratio
    else:
        o2["policy"][:, LA] = PREV_A.clamp(-5.0, 5.0)   # the env clips this obs term to +-5
    return o2


class A:
    pass


a = A()
for k, v in dict(task=TASK, num_envs=N, checkpoint=None, device="cuda:0", distributed=False,
                 seed=5, max_iterations=None, resume=False, run_name=None, logger=None,
                 load_run=None, load_checkpoint=None, video=False, video_length=200).items():
    setattr(a, k, v)
agent_cfg = cli_args.parse_rsl_rl_cfg(TASK, a)
env.clip_actions = None if RAW else 6.0
teacher_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
teacher_runner.load(TEACHER, load_optimizer=False)
teacher = teacher_runner.alg.policy
student_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
student_runner.load(TEACHER, load_optimizer=False)
student = student_runner.alg.policy

# rescale the student's last layer: a_new = a_old * old_scale / new_scale
last = [m for m in student.actor.modules() if isinstance(m, torch.nn.Linear)][-1]
k = (old_sc / new_sc).unsqueeze(1)
with torch.no_grad():
    last.weight.mul_(k)
    last.bias.mul_(k.squeeze(1))
for p in teacher.parameters():
    p.requires_grad_(False)
opt = torch.optim.Adam(student.actor.parameters(), lr=LR)

if os.environ.get("TEACHER_ONLY"):
    # No student: execute the teacher's commanded angle, clipped to the joint window,
    # in the new action space. If this cannot climb, clipping alone changes the
    # dynamics (torque-limited joints get less drive from a clipped target) and
    # distillation cannot work as designed.
    done_n = comp_n = base_n = 0
    obs = env.get_observations()
    prev_a = None
    with torch.inference_mode():
        for step in range(int(os.environ.get("TEACHER_ONLY_STEPS", "3000"))):
            a_t = teacher.act_inference(teacher_obs(obs))
            if step in (2, 3, 4, 30):
                d_ = (teacher_obs(obs)["policy"][:, LA] - obs["policy"][:, LA]).abs()
                print(f"[D] dbg2 step {step}: substituted last-action vs env's: max {float(d_.max()):.3f} mean {float(d_.mean()):.3f}"
                      f"  |env la| {float(obs['policy'][:, LA].abs().mean()):.2f} |PREV_A| {float(PREV_A.abs().mean()) if PREV_A is not None else -1:.2f}",
                      flush=True)
            PREV_A = a_t.clone()
            tgt = (off + old_sc * a_t).clamp(lo, hi)
            if step in (3, 40) and not RAW and prev_a is not None:
                rec = teacher_obs(obs)["policy"][:, LA]
                per = (rec - prev_a).abs().mean(0)
                print(f"[D] dbg step {step}: teacher's last-action input vs its previous output, mean abs diff per joint "
                      f"{[round(float(x), 2) for x in per]}", flush=True)
                print(f"[D] dbg step {step}: extended clip binds on {100 * float(((off + old_sc * a_t < lo) | (off + old_sc * a_t > hi)).float().mean()):.1f}% of joint-steps", flush=True)
            prev_a = a_t.clone()
            act_env = a_t if RAW else ((tgt - off) / new_sc).clamp(-4.0, 4.0)
            obs, rew, dones, infos = env.step(act_env)
            PREV_A[dones.bool()] = 0.0
            if step in (3, 40) and not RAW:
                pa = at.processed_actions
                print(f"[D] dbg step {step}: |processed - teacher target| per joint max "
                      f"{[round(float((pa[:, j] - tgt[:, j]).abs().max()), 3) for j in range(12)]}", flush=True)
                print(f"[D] dbg step {step}: obs last_action vs applied action max diff "
                      f"{float((obs['policy'][:, LA] - u.action_manager.action).abs().max()):.3f}  "
                      f"|obs la| {float(obs['policy'][:, LA].abs().mean()):.2f}  "
                      f"|a_t(old units)| {float(a_t.abs().mean()):.2f}  "
                      f"|act_env| {float(act_env.abs().mean()):.2f}", flush=True)
            d = u.termination_manager.dones
            done_n += int(d.sum())
            comp_n += int((u.termination_manager.get_term("course_complete") & d).sum())
            base_n += int((u.termination_manager.get_term("base_contact") & d).sum())
    print(f"[D] TEACHER_ONLY clipped-target: episodes {done_n}  complete {comp_n} "
          f"({100.0 * comp_n / max(done_n, 1):.1f}%)  base_contact {base_n} ({100.0 * base_n / max(done_n, 1):.1f}%)",
          flush=True)
    env.close()
    simulation_app.close()
    sys.exit(0)

window_done = window_complete = 0
loss_acc = []
stu_abs = []
tgt_abs = []
clipped = []
obs = env.get_observations()
for step in range(STEPS):
    with torch.no_grad():
        a_t = teacher.act_inference(teacher_obs(obs))
        PREV_A = a_t.clone()
        target = (off + old_sc * a_t).clamp(lo, hi)                    # the angle the teacher commands
        y = ((target - off) / new_sc).clamp(-4.0, 4.0)                 # ... in the student's units
        clipped.append(float(((off + old_sc * a_t < lo - 1e-4) | (off + old_sc * a_t > hi + 1e-4)).float().mean()))
    pred = student.act_inference(obs)
    loss = torch.nn.functional.mse_loss(pred, y)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.actor.parameters(), 1.0)
    opt.step()
    noise = max(0.05, 0.3 * (1.0 - step / STEPS))
    act = (pred.detach() + noise * torch.randn_like(pred)).clamp(-4.0, 4.0)
    # DAgger mixing: the teacher drives with probability beta (1 -> 0 over the first 60 % of steps), so the
    # student first learns on good states and then on the states its own errors produce.
    beta = max(0.0, 1.0 - step / (0.6 * STEPS))
    act = torch.where((torch.rand(act.shape[0], 1, device=act.device) < beta), y, act)
    obs, rew, dones, infos = env.step(act)
    PREV_A[dones.bool()] = 0.0
    loss_acc.append(float(loss))
    stu_abs.append(float(pred.detach().abs().mean()))
    tgt_abs.append(float(y.abs().mean()))
    try:
        done = u.termination_manager.dones
        comp = u.termination_manager.get_term("course_complete")
        window_done += int(done.sum())
        window_complete += int((comp & done).sum())
    except Exception:  # noqa: BLE001
        pass
    if (step + 1) % 250 == 0:
        print(f"[D] step {step + 1:5d} loss {sum(loss_acc[-250:]) / 250:.4f}  "
              f"|a_student| {sum(stu_abs[-250:]) / 250:.2f}  |y_target| {sum(tgt_abs[-250:]) / 250:.2f}  "
              f"teacher-past-window {100 * sum(clipped[-250:]) / 250:.0f}%  "
              f"episodes {window_done} complete {window_complete}", flush=True)
        window_done = window_complete = 0

os.makedirs(os.path.dirname(OUT), exist_ok=True)
try:
    student_runner.save(OUT)
except AttributeError:   # rsl_rl's save() touches logger_type after the file is already written
    pass
print(f"[D] saved {OUT}", flush=True)
print("[D] OK", flush=True)
sys.stdout.flush()
env.close()
simulation_app.close()
