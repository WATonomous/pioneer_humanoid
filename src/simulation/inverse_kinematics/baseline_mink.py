"""mink IK baseline in MuJoCo (task B3), scored with metrics.py on the evaluation set.

Step 1: the simulation loop alone. Hold the arm at a pose and measure how far it drifts.
"""
import numpy as np

from arm_model import LeftArm
from arm_sim import EPISODE_STEPS, Sim
from metrics import episode_metrics, pose_errors, summarize
from sampling import sample_start


def hold_test(sim, q):
    """Start at q and command q for a whole episode. Returns episode_metrics against the start pose."""
    sim.reset(q)
    p_goal, R_goal = sim.tcp_pose()
    e_p, e_R = [], []
    for _ in range(EPISODE_STEPS):
        sim.step(q)
        p, R = sim.tcp_pose()
        ep, eR = pose_errors(p, R, p_goal, R_goal)
        e_p.append(ep)
        e_R.append(eR)
    return episode_metrics(np.array(e_p), np.array(e_R))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    arm = LeftArm()
    poses = [sample_start(arm, rng) for _ in range(50)]
    for gravity_comp in (False, True):
        sim = Sim(gravity_comp)
        s = summarize([hold_test(sim, q) for q in poses])
        print(f"gravity comp {gravity_comp}: strict {s['strict_rate']:.0f}%, "
              f"median drift {s['median_final_pos_err'] * 1000:.2f} mm, "
              f"{np.degrees(s['median_final_rot_err']):.3f} deg")
