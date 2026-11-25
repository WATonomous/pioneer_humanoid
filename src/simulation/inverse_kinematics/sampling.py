"""Goals and start poses for PolicyIK, and the fixed evaluation set (task B1)."""
import sys
import numpy as np

from arm_model import SRC, LeftArm
from pathlib import Path

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.npz"   # next to sampling.py

#----Agree the workspace box, torso box, seeds and file location before generating fixed evaluation set


sys.path.append(str(SRC/"teleop"/"quest_controller_teleop"))
from mink_control import TORSO_CENTRE, TORSO_HALF

# Torso keep-out box (base frame, m), shared with the teleop IK. Approximate: confirm with the team.
TORSO_LO = np.array(TORSO_CENTRE) - np.array(TORSO_HALF)
TORSO_HI = np.array(TORSO_CENTRE) + np.array(TORSO_HALF)
UPPER_ARM_R, FOREARM_R, HAND_R = 0.05, 0.045, 0.04   # capsule radii (m), as in mink_control
CLEARANCE = 0.02                                      # extra safety gap (m)

# Workspace box for goals, in the base frame (m). Placeholder
WORKSPACE_LO = np.array([0.15, -0.10, -0.40])
WORKSPACE_HI = np.array([0.55, 0.45, 0.05])


EVAL_SEED = 12345      # used ONLY for the evaluation set, never for training
TRAIN_SEED = 0     #must use seeds other than EVAL_SEED

def arm_points(arm, q):
    """Shoulder, elbow, wrist and tool point positions (each (3,)) at joint angles q."""
    arm.forward(q)

    def joint_pos(name):
        return arm.data.xanchor[arm.model.joint(name).id].copy()

    return (joint_pos("left_shoulder_pitch"), joint_pos("left_elbow"),
            joint_pos("left_wrist_pitch"), arm.data.site_xpos[arm.tcp_id].copy())


def segment_hits_box(a, b, radius, lo, hi):
    """True if a capsule from point a to point b (radius in m) touches the box lo..hi."""
    points = np.linspace(a, b, 10)
    lo = lo - radius - CLEARANCE
    hi = hi + radius + CLEARANCE
    return np.any(np.all((points > lo) & (points < hi), axis=1))

def segments_too_close(a1, b1, r1, a2, b2, r2):
    """True if two capsules (a1->b1 with radius r1, a2->b2 with radius r2) are closer than r1 + r2 + CLEARANCE."""
    points = np.linspace(a1, b1, 10)
    points2 = np.linspace(a2, b2, 10)
    for p1 in points:
        for p2 in points2:
            if np.linalg.norm(p1 - p2) < r1 + r2 + CLEARANCE:
                return True
    return False

def collision_free(arm, q):
    """True if the arm at joint angles q stays clear of the torso and of itself."""
    shoulder, elbow, wrist, tcp = arm_points(arm, q)
    if segment_hits_box(elbow, wrist, FOREARM_R, TORSO_LO, TORSO_HI):
        return False
    if segment_hits_box(wrist, tcp, HAND_R, TORSO_LO, TORSO_HI):
        return False
    if segments_too_close(shoulder, elbow, UPPER_ARM_R, wrist, tcp, HAND_R):
        return False
    return True



def too_close(p, R, other_p, other_R, pos_tol=1e-3, rot_tol=np.radians(0.2)):
    """True if goal (p, R) is within pos_tol AND rot_tol of any goal in other_p (M, 3), other_R (M, 3, 3)."""
    d = np.linalg.norm(other_p - p, axis=1)
    close = np.flatnonzero(d < pos_tol)
    for j in close:
        angle = np.arccos(np.clip((np.trace(R.T @ other_R[j]) - 1) / 2, -1.0, 1.0))
        if angle < rot_tol:
            return True
    return False

def count_too_close(eval_set, n_train=10_000, seed=TRAIN_SEED):
    """How many eval goals are too close to one of n_train training goals. Should be 0."""
    arm = LeftArm()
    rng = np.random.default_rng(seed)
    train_p = np.zeros((n_train, 3))
    train_R = np.zeros((n_train, 3, 3))
    
    for i in range(n_train):
        _, train_p[i], train_R[i] = sample_goal(arm, rng)
    count = 0
    for i in range(len(eval_set["p_goal"])):
        if too_close(eval_set["p_goal"][i], eval_set["R_goal"][i], train_p, train_R):
            count += 1
    return count



def in_workspace(p):
    """True if the tool point position p (3,) is inside the workspace box."""
    return np.all(p>= WORKSPACE_LO) and np.all(p<= WORKSPACE_HI)


def sample_goal(arm, rng):
    """A random reachable goal inside the workspace. Returns (q, p, R)."""
    while True:
        q = arm.random_q(rng)
        if not collision_free(arm, q):
            continue
        p, R = arm.tcp_pose(q)
        if in_workspace(p):
            return q, p, R



def sample_start(arm, rng):
    """Random collision-free joint angles to start from (not limited to the workspace)."""
    while True:
        q = arm.random_q(rng)
        if collision_free(arm, q):
            return q


def make_eval_set(n, seed, path):
    """Make n start/goal pairs from this seed and save them to path (.npz)."""
    arm = LeftArm()
    rng = np.random.default_rng(seed)
    q_start = np.zeros((n, 7))
    q_goal = np.zeros((n, 7))
    p_goal = np.zeros((n, 3))
    R_goal = np.zeros((n, 3, 3))

    for i in range(n):
        q_start[i] = sample_start(arm, rng)
        q_goal[i], p_goal[i], R_goal[i] = sample_goal(arm, rng)

    np.savez(path, q_start=q_start, q_goal=q_goal, p_goal=p_goal, R_goal=R_goal, seed=seed)


def load_eval_set(path):
    """Load a set saved by make_eval_set, as a dict: {"q_start": ..., "q_goal": ..., ...}."""
    data = np.load(path)
    return {key: data[key] for key in data.files}


if __name__ == "__main__":
    make_eval_set(1000, EVAL_SEED, EVAL_SET_PATH)
    s = load_eval_set(EVAL_SET_PATH)
    print("q_start", s["q_start"].shape, "p_goal", s["p_goal"].shape, "R_goal", s["R_goal"].shape)
    print("first goal position:", np.round(s["p_goal"][0], 3))
    print("eval goals too close to a training goal:", count_too_close(s))

