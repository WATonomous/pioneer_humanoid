# actuator_sysid: leg motor bench tools (#335)

Bench software for the CubeMars leg motors (AKH70-48, AK60-39: AK 3.0; AK80-64: AK 2.0), MIT mode.
Motor limits come from `src/pioneer_humanoid/pioneer_humanoid/leg_params.py` (`MotorSpec`).

**Real hardware is locked** until the Task 10 reviewer checkpoint (D11): `bus.LIVE_HARDWARE_ENABLED = False`
refuses every bus except python-can `virtual` and Linux `vcan*`. Unlock = one reviewed commit flipping
that flag; live runs then also need `--live` and typing the motor id. Hardware E-stop on the 48 V
supply, in reach, for every powered test: software can't stop a motor if the PC hangs.

## Setup (host)
```
conda create -n sysid python=3.11 -y && conda activate sysid
pip install python-can==4.6.1 numpy matplotlib pytest
pip install --no-deps -e src/pioneer_humanoid -e tools/actuator_sysid
```
A separate env because python-can needs `packaging>=23.1` and `watonomous_env` pins 23.0 (isaacsim-core).

## Modules
| file | what |
|---|---|
| `protocol.py` | MIT codecs, both protocols (bytes only); motor side for the fake |
| `bus.py` | python-can wrapper: hardware lock, echo of sent frames, kernel timestamps on SocketCAN |
| `motors.py` | `Ak3Motor` / `Ak2Motor`: which frames, which replies (never send) |
| `safety.py` | the only sender: bench limits under the `MotorSpec` ceiling, command + feedback checks, 50 ms watchdog, seed from live position, damp 1 s then disable on any exit |
| `fake_motor.py` | 1-DOF fake motor on a virtual / vcan bus (tests, dry runs) |
| `profiles.py` | step, chirp, speed_sweep, torque_ramp, torque_hold, backlash; `run()` through `Safety` |
| `logger.py` | 500 Hz `Ticker`, `Recorder`, run folders `outputs/actuator_sysid/<motor>_<id>_<profile>_<n>/` (`log.npz` + `meta.json`) |
| `run_profile.py` | CLI: one profile on one motor |
| `bringup.py` | CLI: read-only → kp = 0 (zeroing, 90° hand turn) → low-kp hold |
| `plot_log.py` | CLI: `plot.png` per run folder |

## Dry runs (no hardware)
```
python -m actuator_sysid.bringup --motor AK80-64 --id 50 --fake --fixture "dry run"
python -m actuator_sysid.run_profile --motor AK60-39 --id 40 --profile step --fake --fixture "dry run"
python -m actuator_sysid.plot_log outputs/actuator_sysid/AK60-39_40_step_1
```
`--set name=value` changes a profile parameter; `--limit name=value --limit-reason "..."` changes a
bench limit (raising one above its default needs the reason; it can never exceed the motor's ceiling).

## Tests
```
python -m pytest tools/actuator_sysid/tests -p no:cacheprovider
```
The vcan tests (kernel timestamps) skip unless `vcan0` exists. To create it (needs sudo, lost on reboot):
```
sudo modprobe vcan                     # load the virtual CAN driver
sudo ip link add dev vcan0 type vcan   # create the device
sudo ip link set up vcan0              # switch it on
```
