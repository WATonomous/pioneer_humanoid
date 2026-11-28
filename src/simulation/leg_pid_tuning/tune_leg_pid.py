import os, time
import mujoco, mujoco.viewer, numpy as np, matplotlib.pyplot as plt
from collections import deque

XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leg.xml")
m = mujoco.MjModel.from_xml_path(XML_PATH)
d = mujoco.MjData(m)

Kp, Kd = 267.0, 17.0
tau_stall, v_max = 222.0, 3.67      # AKH70-48
delay_steps = 10                    # 10 ms at 1 kHz
target = -0.5                       # rad step (knee flexes negative; limits [-1.047, 0.087])
duration = 2.0                      # s
slowdown = 5.0                      # >1 plays the viewer in slow motion

buf = deque([0.0] * (delay_steps + 1), maxlen=delay_steps + 1)
log = []

with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    while viewer.is_running() and d.time < duration:
        buf.append(target)
        q_ref = buf[0]                  # delayed command
        q, qd = d.qpos[0], d.qvel[0]

        tau = Kp * (q_ref - q) - Kd * qd

        # torque-speed curve (same as DCMotorCfg)
        tau_max = np.clip(tau_stall * (1 - qd / v_max), 0, tau_stall)
        tau_min = np.clip(tau_stall * (-1 - qd / v_max), -tau_stall, 0)
        tau = np.clip(tau, tau_min, tau_max)

        d.ctrl[0] = tau
        mujoco.mj_step(m, d)
        log.append((d.time, q, qd, tau))

        viewer.sync()
        # pace to wall clock so the motion plays in real time
        lag = d.time * slowdown - (time.time() - start)
        if lag > 0:
            time.sleep(lag)

    # hold the final pose until the window is closed
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.02)

t, q, qd, tau = np.array(log).T
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(t, q); ax[0].axhline(target, ls="--"); ax[0].set_ylabel("angle (rad)")
ax[1].plot(t, qd); ax[1].set_ylabel("speed (rad/s)")
ax[2].plot(t, tau); ax[2].set_ylabel("torque (N·m)"); ax[2].set_xlabel("time (s)")
plt.show()
