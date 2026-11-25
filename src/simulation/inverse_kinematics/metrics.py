"""Success metrics for PolicyIK (task B2), shared by the mink baseline and the policy.

Definitions follow the paper's simulation evaluation: errors are averaged over the final second.
"""
import numpy as np

CONTROL_DT = 1 / 30                 # s per control step (30 Hz)
WINDOW_S = 1.0                      # the final window that is scored (s)
STRICT_POS = 1e-3                   # 1 mm
STRICT_ROT = np.radians(0.2)        # 0.2°
COARSE_POS = 1e-2                   # 1 cm
COARSE_ROT = np.radians(2.0)        # 2°


def rotation_angle(R_a, R_b):
    """Angle (rad) of the rotation between orientations R_a and R_b (each 3x3)."""
    angle = np.arccos(np.clip((np.trace(R_a.T @ R_b) - 1) / 2, -1.0, 1.0))
    return angle


def pose_errors(p, R, p_goal, R_goal):
    """Position error (m) and orientation error (rad) of pose (p, R) from the goal (p_goal, R_goal)."""
    return np.linalg.norm(p - p_goal), rotation_angle(R, R_goal)


def episode_metrics(e_p, e_R, dt=CONTROL_DT, window_s=WINDOW_S):
    """Score one episode from its per-step errors.

    e_p, e_R: arrays (T,), one value per control step; sample i is recorded after step i+1,
    i.e. at time (i + 1) * dt.
    Returns a dict with: strict, coarse, held (bool), entry_time (s, or nan), final_pos_err (m),
    final_rot_err (rad).
    """
    e_p, e_R = np.asarray(e_p), np.asarray(e_R)
    n = round(window_s / dt)           # samples in the window: 30
    p_win, R_win = e_p[-n:], e_R[-n:]
    final_pos_err = np.mean(p_win)
    final_rot_err = np.mean(R_win)
    strict = final_pos_err <= STRICT_POS and final_rot_err <= STRICT_ROT
    held = np.all(p_win <= STRICT_POS) and np.all(R_win <= STRICT_ROT)
    entry_time = np.nan
    idx = np.flatnonzero((e_p <= STRICT_POS) & (e_R<= STRICT_ROT))
    if len(idx) > 0:
        entry_time = (idx[0] + 1) * dt
    coarse = final_pos_err <= COARSE_POS and final_rot_err <= COARSE_ROT
    return dict(strict=strict, held=held, entry_time=entry_time,
                final_pos_err=final_pos_err, final_rot_err=final_rot_err, coarse=coarse)


def summarize(results):
    """Combine many episode_metrics dicts into rates (%) and medians, over ALL tasks."""
    strict_rate = np.mean([r["strict"] for r in results]) * 100
    coarse_rate = np.mean([r["coarse"] for r in results]) * 100
    held_rate = np.mean([r["held"] for r in results]) * 100
    median_entry_time = np.nanmedian([r["entry_time"] for r in results])
    median_final_pos_err = np.median([r["final_pos_err"] for r in results])
    median_final_rot_err = np.median([r["final_rot_err"] for r in results])
    n_tasks = len(results)
    return dict(strict_rate=strict_rate, held_rate=held_rate,
                median_entry_time=median_entry_time,
                median_final_pos_err=median_final_pos_err,
                median_final_rot_err=median_final_rot_err,
                n_tasks=n_tasks, coarse_rate=coarse_rate)