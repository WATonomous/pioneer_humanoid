"""Is the leg buzz in the contact, or in the joint?

Every command-side explanation has failed: pricing the knee's magnitude did
nothing, and capping the target's slew 20x (target reversals 9.8 -> 0.5/s)
barely moved the knee joint (16.6 -> 11.7 reversals/s) while destroying the
climb. So the target is not what shakes.

There is also a confound in the old reversal count: the spiral knee sits pinned
near zero velocity for ~48 steps at a time, where numerical noise flips the
sign. That counts time-spent-still, not shake.

This probe measures the two remaining candidates directly:

* contact chatter -- how often the stance foot's force collapses and rebounds
  while the foot is still planted, plus contact on/off transitions,
* sole buzz -- the high-frequency part of the foot body's vertical motion while
  planted (3-point residual, in mm),
* joint reversals with a deadband, so a flip only counts when the velocity
  actually moves on both sides of it, reported separately for stance and swing.

Run on the spiral checkpoint and the straight deliverable for comparison.
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
from humanoid_rl_tasks.locomotion.config.pioneer import gait as gait_mdp  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import cli_args  # noqa: E402

TASK = os.environ.get("TASK_OVERRIDE", "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-Play-v0")
CKPT = os.environ.get("CKPT_OVERRIDE",
                      "/workspace/humanoid/outputs/rl/pioneer_humanoid_spiral_foothold/2026-11-16_22-26-35/model_299.pt")
N, STEPS, WARMUP = (int(os.environ.get("BUZZ_N", "4")),
                    int(os.environ.get("BUZZ_STEPS", "900")), 60)
CONTACT_N = 20.0      # force above which the foot counts as planted
DEADBAND = 0.30       # rad/s of knee speed a reversal must exceed on both sides

# Four spawn slots, cycled to fill however many envs the sample uses: the tap
# rate varied 3x between two runs of the same checkpoint at N=4, so the numbers
# need a wider sample before they can be compared across runs.
_BASE_SLOTS = (
    (1.551, -0.777, 0.816, 1.5708), (0.213, 1.500, 1.417, 3.0005),
    (-1.504, -0.182, 2.017, 4.8331), (1.361, -0.666, 2.857, 7.3987),
)
SLOTS = tuple(_BASE_SLOTS[i % len(_BASE_SLOTS)] for i in range(N))

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
cfg.scene.env_spacing = 0.0
cfg.observations.policy.enable_corruption = False
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg))
u = env.unwrapped
if "Spiral" in TASK:
    term = u.event_manager.get_term_cfg("reset_base")
    term.params["slots"] = SLOTS
    term.params["xy_jitter"] = 0.03
    term.params["yaw_jitter"] = 0.03


class A:
    pass


a = A()
for k, v in dict(task=TASK, num_envs=N, checkpoint=None, device="cuda:0", distributed=False,
                 seed=5, max_iterations=None, resume=False, run_name=None, logger=None,
                 load_run=None, load_checkpoint=None, video=False, video_length=200).items():
    setattr(a, k, v)
agent_cfg = cli_args.parse_rsl_rl_cfg(TASK, a)
env.clip_actions = agent_cfg.clip_actions   # the runner cfg's bound; the wrapper defaults to none
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(CKPT, load_optimizer=False)
policy = runner.get_inference_policy(device=u.device)

robot = u.scene["robot"]
sensor = u.scene["contact_forces"]
dt = u.step_dt

feet = ["l_foot", "r_foot"]
# The sensor covers every robot body, so its body order need not match the
# robot's: resolve the ids in the *sensor's* frame or the force reads zero.
feet_cfg = SceneEntityCfg("contact_forces", body_names=feet, preserve_order=True)
feet_cfg.resolve(u.scene)
fid = list(feet_cfg.body_ids)
# Positions come from the robot's own body array, so they need the robot's ids.
rid = [robot.body_names.index(f) for f in feet]
print(f"[B] sensor bodies {len(sensor.body_names)}  feet ids {fid}  "
      f"names {[sensor.body_names[i] for i in fid]}  robot ids {rid}")
knees = ["Knee_L", "Knee_R"]
kid = [robot.joint_names.index(k) for k in knees]
leg_idx = [i for i, n in enumerate(robot.joint_names)
           if any(k in n.lower() for k in ("hip", "knee", "ankle"))]

F, Z, V, BZ, Q = [], [], [], [], []
MISM, MISM_HEAVY, HEAVY = [], [], []
CLOCK_PERIOD = float(os.environ.get("SPIRAL_CLOCK_PERIOD", "3.0"))
feet_cfg_clock = SceneEntityCfg("contact_forces", body_names=feet, preserve_order=True)
feet_cfg_clock.resolve(u.scene)
obs = env.get_observations()
with torch.inference_mode():
    for step in range(STEPS + WARMUP):
        act = policy(obs)
        obs, *_ = env.step(act)
        if step >= WARMUP:
            f = sensor.data.net_forces_w[:, fid, :].norm(dim=-1)
            F.append(f.clone())
            Z.append(robot.data.body_pos_w[:, rid, 2].clone())
            V.append(robot.data.joint_vel[:, kid].clone())
            Q.append(robot.data.joint_pos[:, kid].clone())
            BZ.append(robot.data.root_pos_w[:, 2].clone())
            # The gait prior's own charge, and the same charge if "stance"
            # meant "bearing weight" instead of "touching".
            MISM.append(gait_mdp.contact_mismatch_l2(u, feet_cfg_clock, CLOCK_PERIOD))
            phase = gait_mdp.leg_phase(u, CLOCK_PERIOD)
            heavy = (f > CONTACT_N)
            m = (heavy == (phase < gait_mdp.STANCE)).float().sum(dim=1)
            MISM_HEAVY.append((2.0 - m).square())
            HEAVY.append(heavy)

F = torch.stack(F)   # (T, N, 2)
Z = torch.stack(Z)   # (T, N, 2)
V = torch.stack(V)   # (T, N, 2)
Q = torch.stack(Q)   # (T, N, 2)
BZ = torch.stack(BZ)  # (T, N)
T = F.shape[0]

planted = F > CONTACT_N
print(f"[B] {os.path.basename(CKPT)}  {T} steps @ {1.0/dt:.0f} Hz  "
      f"planted frac {float(planted.float().mean()):.2f}  "
      f"force mean {float(F.mean()):.0f} N max {float(F.max()):.0f} N")

# --- contact chatter -------------------------------------------------------
# A transition is a planted->air or air->planted edge. A real gait gives ~2 per
# stride per foot; a buzzing contact gives many per second.
edges = (planted[1:] != planted[:-1]).float().mean()
print(f"[B] contact transitions      {float(edges) / dt:7.2f} /s   "
      f"(a 1 s stride gives ~2)")

# How long is each airborne gap, and -- the part that decides whether a "tap"
# is real -- how far the sole actually rises off the tread during it? A one-step
# gap with a 0.3 mm rise is contact-force chatter; a 20 mm rise is a foot that
# leaves the stair and comes back.
gaps = []
gaps_rise = []
for e in range(N):
    for f_ in range(2):
        air = ~planted[:, e, f_]
        zf = Z[:, e, f_]
        run = 0
        last_planted_z = None
        for i, flag in enumerate(air.tolist()):
            if not flag:
                last_planted_z = float(zf[i])
            if flag:
                run += 1
            elif run:
                gaps.append(run * dt)
                gaps_rise.append(1000.0 * (float(zf[i - run:i].max()) - (last_planted_z if last_planted_z is not None else float(zf[i - run]))))
                run = 0
        if run:
            gaps.append(run * dt)
            gaps_rise.append(1000.0 * (float(zf[-run:].max()) - (last_planted_z if last_planted_z is not None else float(zf[-run]))))
if gaps:
    gaps.sort()
    short = sum(1 for g in gaps if g < 0.10)
    print(f"[B] airborne gaps            {len(gaps):4d} total  "
          f"median {gaps[len(gaps) // 2]:.3f} s  "
          f"<0.10 s: {short} ({100.0 * short / len(gaps):.0f}%)  "
          f"-> {short / (T * N * 2 * dt):5.2f} taps/s")
    rises = sorted(r for g, r in zip(gaps, gaps_rise) if g < 0.10)
    if rises:
        n_r = len(rises)
        print(f"[B] sole rise during taps    p50 {rises[n_r // 2]:6.2f} mm  "
              f"p90 {rises[int(0.9 * n_r)]:6.2f} mm  max {rises[-1]:6.2f} mm  "
              f"(a real hop clears >10 mm)")
    long_ = len(gaps) - short
    z_rate = float(((BZ[1:] - BZ[:-1]).clamp_min(0) / dt).mean())
    print(f"[B] climb                    {z_rate:5.3f} m/s vertical  "
          f"{z_rate / 0.12:5.2f} risers/s  "
          f"{z_rate / 0.12 / max(long_ / (T * N * 2 * dt), 1e-6):5.2f} risers per real swing")
    print(f"[B] air episodes >=0.10 s    {long_:4d}  "
          f"({long_ / (T * N * 2 * dt):5.2f} /s per foot)")
    taps_s = short / (T * N * 2 * dt)
    risers_s = z_rate / 0.12
    print(f"[B] taps per riser           {taps_s / max(risers_s, 1e-6):5.2f}  "
          f"(straight deliverable ~0.4)")

# While planted, does the force collapse and come back? Compare against a
# 5-step local mean so a real unload (lift-off) is not counted.
Fp = F.permute(1, 2, 0).reshape(N * 2, 1, T)
loc = torch.nn.functional.avg_pool1d(
    torch.nn.functional.pad(Fp, (2, 2), mode="replicate"), kernel_size=5, stride=1
).reshape(N, 2, T).permute(2, 0, 1)
dip = (F < 0.5 * loc) & planted & (loc > CONTACT_N)
print(f"[B] force dips <50% of local mean (planted)    {float(dip.float().sum()) / (T * N * 2 * dt):7.2f} /s")
dF = (F[1:] - F[:-1]).abs() / loc[1:].clamp_min(1.0)
both = planted[1:] & planted[:-1]
print(f"[B] relative force change while planted        {float(dF[both].mean()) if bool(both.any()) else 0.0:7.3f} /step")

# --- sole buzz -------------------------------------------------------------
# 3-point residual of the foot's z while planted: how many mm the sole moves
# back and forth between samples, above whatever the whole-body motion does.
res = (Z[2:] - 2 * Z[1:-1] + Z[:-2]).abs() * 1000.0
st = planted[1:-1]
print(f"[B] sole z 3-point residual while planted      {float(res[st].mean()) if bool(st.any()) else 0.0:7.3f} mm")
# reference: the same residual on the base body, i.e. whole-body motion
bres = (BZ[2:] - 2 * BZ[1:-1] + BZ[:-2]).abs() * 1000.0
print(f"[B] base z 3-point residual (reference)        {float(bres.mean()):7.3f} mm")

# --- how much does the knee angle actually shake? --------------------------
# The visible jitter is the joint angle oscillating faster than the gait. Strip
# a 9-step (~0.18 s) moving average -- slower than any real knee motion inside a
# swing -- and report what is left, as RMS in mrad. A smooth gait's knee moves
# monotonically through a swing, so its residual is a few mrad; a limit cycle
# leaves tens.
for i, name in enumerate(knees):
    q = Q[:, :, i]
    qp = torch.nn.functional.avg_pool1d(
        torch.nn.functional.pad(q.T.unsqueeze(1), (4, 4), mode="replicate"),
        kernel_size=9, stride=1).squeeze(1).T
    hf = (q - qp) * 1000.0
    print(f"[B] {name:8s} knee hf residual   rms {float(hf.pow(2).mean().sqrt()):7.1f} mrad  "
          f"p99 {float(hf.abs().flatten().quantile(0.99)):7.1f} mrad  "
          f"= {float(hf.pow(2).mean().sqrt()) / 10.0:5.2f} deg")

# --- can the gait clock see the rattle? ------------------------------------
light = (F > 1.0) & (F <= CONTACT_N)
mism = torch.stack(MISM)
mism_heavy = torch.stack(MISM_HEAVY)
heavy = torch.stack(HEAVY)
print(f"[B] light contact 1-{CONTACT_N:.0f} N (touching, not bearing)   {float(light.float().mean()):7.2f} of steps")
print(f"[B] stance share  touch>1N {float((F > 1.0).float().mean()):.2f}   "
      f"load>{CONTACT_N:.0f}N {float(heavy.float().mean()):.2f}")
print(f"[B] contact_mismatch charge  as coded (1 N) {float(mism.mean()):7.3f}   "
      f"if stance meant >{CONTACT_N:.0f} N {float(mism_heavy.mean()):7.3f}")

# --- joint reversals with a deadband ---------------------------------------
def rev(x, mask):
    d = x[1:] - x[:-1]
    s = torch.sign(d)
    prev = d[:-1]
    nxt = d[1:]
    flip = (s[1:] * s[:-1] < 0) & (prev.abs() > DEADBAND * dt) & (nxt.abs() > DEADBAND * dt)
    m = mask[1:-1]
    if not bool(m.any()):
        return 0.0
    return float(flip[m].float().sum()) / (float(m.float().sum()) * dt)


for i, name in enumerate(knees):
    v = V[:, :, i]
    print(f"[B] {name:8s} reversals  all {rev(v, torch.ones_like(v, dtype=torch.bool)):6.2f} /s   "
          f"planted {rev(v, planted[:, :, i]):6.2f} /s   "
          f"airborne {rev(v, ~planted[:, :, i]):6.2f} /s   "
          f"|v| mean {float(v.abs().mean()):5.2f} rad/s")

print("[B] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
