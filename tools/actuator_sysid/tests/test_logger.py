"""logger.py, run_profile.py and plot_log.py: run folder naming, .npz/.json round trip, plots."""
import json
import re
import time

import numpy as np
import pytest

from actuator_sysid import plot_log, run_profile
from actuator_sysid.bus import is_virtual_can
from actuator_sysid.fake_motor import FakeMotor
from actuator_sysid.logger import Ticker, git_state, load_run, next_run_dir
from actuator_sysid.motors import make_motor
from actuator_sysid.protocol import CanFrame
from pioneer_humanoid.leg_params import AK60_39


# --- Run folders ----------------------------------------------------------------------------
def test_run_dirs_count_up_per_motor_id_and_profile(tmp_path):
    names = [next_run_dir(tmp_path, "AK60-39", 40, "step").name for _ in range(3)]
    assert names == ["AK60-39_40_step_1", "AK60-39_40_step_2", "AK60-39_40_step_3"]
    assert next_run_dir(tmp_path, "AK60-39", 41, "step").name == "AK60-39_41_step_1"
    assert next_run_dir(tmp_path, "AK60-39", 40, "chirp").name == "AK60-39_40_chirp_1"
    (tmp_path / "AK60-39_40_step_9").mkdir()
    (tmp_path / "AK60-39_40_step_x").mkdir()  # not a number: ignored
    assert next_run_dir(tmp_path, "AK60-39", 40, "step").name == "AK60-39_40_step_10"


def test_ticker_keeps_rate_and_skips_after_a_stall():
    t = Ticker(0.002)
    for i in range(250):
        t.wait()
        if i == 100:
            time.sleep(0.011)  # stall: ~5 periods
    s = t.stats()
    assert s["ticks"] == 250 and 4 <= s["missed"] <= 6 and s["max_late_s"] >= 0.009
    assert t.elapsed == pytest.approx((250 - 1 + s["missed"]) * 0.002)


def test_git_state_reports_the_commit():
    g = git_state()
    assert re.fullmatch(r"[0-9a-f]{40}", g["commit"]) and isinstance(g["dirty"], bool)


# --- run_profile end to end -----------------------------------------------------------------------
def _run(tmp_path, channel, *extra, motor="AK60-39", profile="step"):
    argv = ["--motor", motor, "--id", "40", "--profile", profile, "--channel", channel, "--fixture", "pytest",
            "--firmware", "fw-test", "--voltage", "48", "--yes", "--out", str(tmp_path), *extra]
    return run_profile.main(argv)


@pytest.fixture
def step_run(tmp_path, channel):
    code = _run(tmp_path, channel, "--fake", "--set", "hold=0.2", "--set", "repeats=1", "--limit", "damp_time=0.2")
    return code, tmp_path / "AK60-39_40_step_1"


def test_run_profile_writes_npz_and_json(step_run):
    code, run_dir = step_run
    assert code == 0 and sorted(p.name for p in run_dir.iterdir()) == ["log.npz", "meta.json"]
    a, meta = load_run(run_dir)
    for key in ("motor", "firmware", "fixture", "gains", "voltage_v", "temperatures_c", "git", "limits", "profile",
                "bus", "events", "result", "loop", "start_position"):
        assert key in meta, key
    assert meta["motor"]["short"] == "AK60-39" and meta["motor"]["id"] == 40 and meta["firmware"] == "fw-test"
    assert meta["gains"] == {"kp": 5.0, "kd": 0.5} and meta["voltage_v"] == 48.0
    assert meta["dry_run"] is True and meta["bus"]["kind"] == "virtual"
    assert meta["result"] == {"ended": "done", "fault": None, "disable_acknowledged": True, "exit_code": 0}
    assert meta["temperatures_c"] == {"start": 30, "end": 30, "max": 30}
    assert meta["start_position"] == pytest.approx(0.3, abs=2e-3)  # --fake-position default
    # Counts in the metadata match the arrays.
    assert len(a["frame_t"]) == meta["counts"]["frames"] and len(a["cmd_t"]) == meta["counts"]["commands"]
    assert a["frame_data"].shape == (len(a["frame_t"]), 8)
    kinds = set(a["cmd_kind"])
    assert {"cmd", "damp"} <= kinds
    assert len(a["tick_t"]) >= 0.9 * 1.0 / 0.002  # 5 holds of 0.2 s at 500 Hz
    assert np.all(np.diff(a["fb_t"]) >= 0)


def test_raw_frames_re_decode_to_the_logged_feedback(step_run):
    _, run_dir = step_run
    a, _ = load_run(run_dir)
    motor = make_motor(AK60_39, 40)
    decoded = []
    for ts, fid, ext, echo, dlc, data in zip(a["frame_t"], a["frame_id"], a["frame_ext"], a["frame_echo"],
                                             a["frame_dlc"], a["frame_data"]):
        fb = None if echo else motor.parse(CanFrame(int(fid), bytes(data[:dlc]), bool(ext)))
        if fb is not None:
            decoded.append((ts, fb.position, fb.torque))
    d = np.array(decoded)
    assert len(d) == len(a["fb_t"])
    np.testing.assert_array_equal(d[:, 0], a["fb_t"])
    np.testing.assert_allclose(d[:, 1], a["fb_position"])
    np.testing.assert_allclose(d[:, 2], a["fb_torque"])


def test_echoes_are_logged_for_every_sent_frame(step_run):
    _, run_dir = step_run
    a, _ = load_run(run_dir)
    # Every command (cmd + damp) plus the disable frames came back as an echo.
    assert a["frame_echo"].sum() >= len(a["cmd_t"])


def test_fault_is_saved_with_exit_code_2(tmp_path, channel):
    with FakeMotor(AK60_39, 40, channel=channel, position=0.1) as fake:
        fake.temperature = 80  # over the 70 degC bench limit
        code = _run(tmp_path, channel, "--set", "hold=0.2", "--limit", "damp_time=0.1")
    assert code == 2
    _, meta = load_run(tmp_path / "AK60-39_40_step_1")
    assert "temperature" in meta["result"]["fault"] and meta["result"]["exit_code"] == 2
    assert any("FAULT" in e["text"] for e in meta["events"])


def test_ak2_motor_runs_too(tmp_path, channel):
    code = _run(tmp_path, channel, "--fake", "--set", "peak=0.5", "--set", "rate=5", "--set", "lead=0.1",
                "--limit", "damp_time=0.1", motor="AK80-64", profile="torque_ramp")
    assert code == 0
    _, meta = load_run(tmp_path / "AK80-64_40_torque_ramp_1")
    assert meta["motor"]["protocol"] == "ak2" and meta["result"]["disable_acknowledged"] is None


@pytest.mark.parametrize("extra,match", [
    (["--limit", "max_torque=10"], "reason"),  # raising a cap without a reason
    (["--limit", "max_torque=100", "--limit-reason", "x"], "ceiling"),  # above the AK60-39 peak 72
    (["--set", "size=0.4"], "jumps"),  # step bigger than the tracking limit
    (["--interface", "socketcan", "--channel", "can0"], "D11"),  # real hardware is locked
    (["--live", "--interface", "socketcan", "--channel", "can0"], "--yes is not allowed"),
])
def test_refused_before_anything_is_sent(tmp_path, channel, capsys, extra, match):
    assert _run(tmp_path, channel, *extra) == 3
    assert match in capsys.readouterr().err
    assert not any(tmp_path.iterdir())  # no run folder either


# --- plot_log ---------------------------------------------------------------------------------
def test_plot_on_fake_data(step_run):
    _, run_dir = step_run
    assert plot_log.main([str(run_dir)]) == 0
    png = run_dir / "plot.png"
    assert png.exists() and png.stat().st_size > 20_000
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_handles_a_run_with_no_feedback(tmp_path, channel, capsys):
    code = _run(tmp_path, channel, "--limit", "damp_time=0.1")  # nobody answers: AK 3.0 is never enabled
    assert code == 2
    run_dir = tmp_path / "AK60-39_40_step_1"
    _, meta = load_run(run_dir)
    assert meta["counts"]["commands"] == 0 and "no feedback" in meta["result"]["fault"]
    assert plot_log.main([str(run_dir)]) == 0


def test_model_torque_uses_the_latest_feedback():
    a = {"cmd_t": np.array([1.0, 2.0]), "fb_t": np.array([0.5, 1.5]), "fb_position": np.array([0.0, 0.1]),
         "fb_velocity": np.array([0.0, 1.0]), "cmd_kp": np.array([10.0, 10.0]), "cmd_kd": np.array([1.0, 1.0]),
         "cmd_p": np.array([0.2, 0.2]), "cmd_v": np.array([0.0, 0.0]), "cmd_t_ff": np.array([0.5, 0.0])}
    np.testing.assert_allclose(plot_log.model_torque(a), [10 * 0.2 + 0.5, 10 * 0.1 - 1.0])


def test_meta_json_is_plain_json(step_run):
    _, run_dir = step_run
    json.loads((run_dir / "meta.json").read_text())
    assert not (run_dir / "meta.json.tmp").exists()


# --- Same run on vcan0 (SocketCAN, kernel timestamps); skipped unless vcan0 is up ---------------
@pytest.mark.skipif(not is_virtual_can("vcan0"), reason="vcan0 not set up (needs sudo, see README)")
def test_run_profile_on_vcan_with_kernel_timestamps(tmp_path):
    code = run_profile.main(["--motor", "AK60-39", "--id", "40", "--profile", "step", "--interface", "socketcan",
                             "--channel", "vcan0", "--fake", "--fixture", "pytest vcan", "--yes", "--out", str(tmp_path),
                             "--set", "hold=0.2", "--set", "repeats=1", "--limit", "damp_time=0.2"])
    assert code == 0
    a, meta = load_run(tmp_path / "AK60-39_40_step_1")
    assert meta["bus"] == {"interface": "socketcan", "channel": "vcan0", "kind": "vcan", "timestamp_source": "kernel"}
    assert meta["result"]["disable_acknowledged"] is True
    # The fake's replies come from another local socket: they must not be taken for our echoes.
    assert len(a["fb_t"]) > 500
    echo = a["frame_echo"]
    assert echo.sum() >= len(a["cmd_t"])
    # Each echo's kernel timestamp comes just after the host time taken before send().
    echo_t = a["frame_t"][echo][:len(a["cmd_t"])]
    delay = echo_t - a["cmd_t"]
    assert np.all(delay > 0) and np.median(delay) < 0.001
