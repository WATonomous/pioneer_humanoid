import json
import math
import numpy as np
import mujoco
from articulated_arm import build, set_controls, ROOT, linkage, LIMITS

model, data = build()
assert model.njnt == 22
assert model.nmesh == 474
baseline = {model.body(i).name: data.xpos[i].copy() for i in range(model.nbody)}
checks = []
worst = 0.
for channel in range(14):
    values = np.zeros(14)
    values[channel] = math.radians(20)
    set_controls(model, data, values)
    eq = np.array([abs(data.efc_pos[i]) for i in range(data.nefc)
                   if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY])
    residual = float(eq.max()) if len(eq) else 0.
    worst = max(worst, residual)
    moved = [model.body(i).name for i in range(model.nbody)
             if np.linalg.norm(data.xpos[i]-baseline[model.body(i).name]) > 1e-5]
    # Pure-axis bodies can remain centered while rotating: inspect matrices too.
    selected_side = 'right' if channel < 7 else 'left'
    opposite = 'left' if channel < 7 else 'right'
    assert not any(name.startswith(opposite) for name in moved), moved
    checks.append({'channel': channel, 'side': selected_side, 'label': channel % 7+1,
                   'moved_body_count': len(moved), 'loop_closure_residual_m': residual})
    assert residual < 2e-6, (channel, residual)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.xpos).all()
for angle in np.linspace(-10, 90, 201):
    v = np.zeros(14)
    v[6] = v[13] = math.radians(angle)
    set_controls(model, data, v)
    residual = max([abs(data.efc_pos[i]) for i in range(data.nefc)
                    if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY] or [0.])
    worst = max(worst, float(residual))
    assert residual < 2e-6
set_controls(model, data, np.zeros(14))
report = {'status': 'passed', 'moving_joints': model.njnt, 'independent_keyboard_controls': 14,
          'source_visual_meshes_preserved': model.nmesh, 'per_control_motion_checks': checks,
          'gripper_sweep_samples': 201, 'maximum_closed_loop_residual_m': worst,
          'mode': 'kinematic preview; user-configurable motion ranges, not hardware stop certification'}
(ROOT/'simulation'/'articulation_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
