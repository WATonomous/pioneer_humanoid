"""Which loose objects have fallen off the table, for the teleops: a take can't be finished once one has."""
from __future__ import annotations

import mujoco

TABLE_TOP_Z = 0.705      # table top of every table scene here
BELOW = 0.05             # m under the table top: well off it, not just tipped over its edge


def off_table(model: mujoco.MjModel, data: mujoco.MjData, table_z: float = TABLE_TOP_Z) -> set[str]:
    """Names of free-jointed bodies whose centre is more than BELOW under the table top. Call once after a
    reset for the ones that start there (none in the table scenes), and ignore those."""
    out = set()
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            body = model.jnt_bodyid[j]
            if data.xpos[body][2] < table_z - BELOW:
                out.add(model.body(body).name)
    return out
