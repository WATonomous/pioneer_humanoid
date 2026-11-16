"""Lift retargeted frames whose body (not only the feet) is below the floor. retarget_lafan.py lifts each frame by its
lowest FOOT point; lying and crawling frames have the back, hips or arms lowest, 12-26 cm into the floor, and resetting
1024 simulated robots into that blows up the GPU sim. Here every collision box of the torso, pelvis, head and legs
counts; the arms are left out (Pioneer's long arm reaches lower than the hips, so lifting by the hand left the hips
hovering where the actor sat on the floor; a light arm slightly in the floor is pushed out gently). Rewrites in place.

  python lift_out_of_floor.py <retarget.xml> <csv> [<csv> ...]
"""
import sys

import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path(sys.argv[1])
d = mujoco.MjData(m)
ARM = ("arm", "forearm", "hand")
boxes = [g for g in range(m.ngeom) if m.geom_group[g] == 3 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
         and not any(k in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) for k in ARM)]
corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
for path in sys.argv[2:]:
    q = np.loadtxt(path, delimiter=",")
    lows = []
    for row in q:
        d.qpos[:3], d.qpos[3:7], d.qpos[7:] = row[:3], row[[6, 3, 4, 5]], row[7:]
        mujoco.mj_kinematics(m, d)
        lows.append(min((corners * m.geom_size[g] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min() for g in boxes))
    lift = np.maximum(0.0, -np.array(lows))
    q[:, 2] += np.convolve(np.pad(lift, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    np.savetxt(path, q, delimiter=",", fmt="%.6f")
    print(f"{path}: lifted {100 * (lift > 0.01).mean():.0f}% of frames, max {100 * lift.max():.1f} cm")
