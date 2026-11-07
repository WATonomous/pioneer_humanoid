"""bringup.py dry runs: stage order, nothing sent in read-only, zeroing, operator prompts while armed."""
import math
import threading
import time

import pytest

from actuator_sysid import bringup
from actuator_sysid.fake_motor import FakeMotor
from actuator_sysid.logger import load_run
from pioneer_humanoid.leg_params import AK60_39, AK80_64

FAST = ["--read-time", "0.3", "--kp0-time", "1.5", "--hold-time", "0.5", "--limit", "damp_time=0.1"]


def _argv(tmp_path, channel, motor, *extra):
    return ["--motor", motor, "--id", "40", "--channel", channel, "--fixture", "pytest", "--out", str(tmp_path),
            *FAST, *extra]


def _meta(tmp_path, motor):
    return load_run(tmp_path / f"{motor}_40_bringup_1")


def test_ak3_dry_run_all_stages(tmp_path, channel):
    assert bringup.main(_argv(tmp_path, channel, "AK60-39", "--fake", "--yes")) == 0
    a, meta = _meta(tmp_path, "AK60-39")
    st = meta["stages"]
    assert st["read_only"]["frames"] > 100 and st["read_only"]["sent"] is False
    assert 700 <= st["read_only"]["rate_hz"] <= 1100  # fake streams at 1000 Hz
    assert "zero" not in st  # AK 3.0: only with --zero
    assert set(st) == {"read_only", "kp0", "hold"}
    assert st["hold"]["kp"] == 2.0 and st["hold"]["max_track_error_rad"] < 0.05
    assert meta["result"] == {"fault": None, "disable_acknowledged": True, "exit_code": 0}
    # Commands: zero-torque during kp0, kp 2 during hold, damping at the end. Never anything else.
    cmd = a["cmd_kind"] == "cmd"
    kps = set(round(float(k), 3) for k in a["cmd_kp"][cmd])
    assert kps == {0.0, 2.0}
    assert set(a["cmd_kind"]) == {"cmd", "damp"}  # AK 3.0 needs no probes
    texts = [e["text"] for e in meta["events"]]
    assert texts.index("stage read_only") < next(i for i, t in enumerate(texts) if t.startswith("armed"))
    assert meta["bringup"]["kp0_reason"] and meta["bringup"]["kp0_limits"]["position_window"] == 3.2


def test_ak2_dry_run_enters_zeros_and_exits(tmp_path, channel):
    with FakeMotor(AK80_64, 40, channel=channel, position=0.7) as fake:
        assert bringup.main(_argv(tmp_path, channel, "AK80-64", "--yes")) == 0
    _, meta = _meta(tmp_path, "AK80-64")
    st = meta["stages"]
    assert st["read_only"]["frames"] == 0  # AK 2.0 is silent until commanded
    assert st["zero"]["ok"] and abs(st["zero"]["position_after"]) < 0.01
    kinds = [k for k, _ in fake.received]
    assert kinds[0] == "enter" and "zero" in kinds and kinds[-1] == "exit"
    # Zeroing only happened between zero-torque commands.
    i = kinds.index("zero")
    before = [c for k, c in fake.received[:i] if k == "mit"]
    assert before and all(c.kp == 0 and c.kd == 0 for c in before)


def test_read_only_and_ctrl_c_before_arming_send_nothing(tmp_path, channel):
    answers = iter(["", KeyboardInterrupt()])

    def input_fn(prompt):
        a = next(answers)
        if isinstance(a, BaseException):
            raise a
        return a
    with FakeMotor(AK60_39, 40, channel=channel) as fake:
        assert bringup.main(_argv(tmp_path, channel, "AK60-39"), input_fn=input_fn) == 130
    assert fake.received == []
    _, meta = _meta(tmp_path, "AK60-39")
    assert meta["stages"]["read_only"]["frames"] > 0 and "never armed: nothing sent" in [e["text"] for e in meta["events"]]


def test_hand_turn_reads_90_deg_and_slow_operator_does_not_trip_watchdog(tmp_path, channel):
    """Operator answers take 0.3 s each (6x the watchdog) while armed; a 'hand' turns the fake 90 deg."""
    prompts = []

    def input_fn(prompt):
        prompts.append(prompt)
        time.sleep(0.3)
        return "y" if "(y/n)" in prompt else ""

    with FakeMotor(AK60_39, 40, channel=channel, position=0.2) as fake:
        def hand():
            while not fake.enabled:  # wait for stage 2
                time.sleep(0.01)
            time.sleep(0.3)
            for _ in range(50):  # +90 deg over 0.5 s (0.5 s < kp0 stage 1.5 s)
                fake.theta += math.pi / 2 / 50
                time.sleep(0.01)
        t = threading.Thread(target=hand, daemon=True)
        t.start()
        code = bringup.main(_argv(tmp_path, channel, "AK60-39"), input_fn=input_fn)
        t.join(1)
    assert code == 0
    _, meta = _meta(tmp_path, "AK60-39")
    kp0 = meta["stages"]["kp0"]
    assert kp0["max_abs_delta_deg"] == pytest.approx(90, abs=1.0)
    assert kp0["final_delta_deg"] == pytest.approx(90, abs=1.0)  # positive: same sign as the turn
    assert kp0["operator_90deg_reads_90"] == "y" and kp0["operator_sign_ok"] == "y"
    assert meta["stages"]["hold"]["operator_ok"] == "y"
    assert meta["result"]["fault"] is None
    assert len(prompts) == 6


def test_ak3_without_feedback_stops_after_read_only(tmp_path, channel):
    assert bringup.main(_argv(tmp_path, channel, "AK60-39", "--yes")) == 2  # nobody on the bus
    _, meta = _meta(tmp_path, "AK60-39")
    assert meta["stages"]["stopped"] == "no feedback in read-only" and meta["result"]["exit_code"] == 2
    assert meta["counts"]["commands"] == 0


@pytest.mark.parametrize("extra,match", [
    (["--interface", "socketcan", "--channel", "can0"], "D11"),
    (["--live", "--interface", "socketcan", "--channel", "can0", "--yes"], "--yes is not allowed"),
    (["--hold-kp", "20"], "kp"),
    (["--limit", "max_speed=2"], "reason"),
])
def test_refused_before_the_bus_opens(tmp_path, channel, capsys, extra, match):
    assert bringup.main(_argv(tmp_path, channel, "AK60-39", *extra)) == 3
    assert match in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_fault_during_hold_is_saved(tmp_path, channel):
    with FakeMotor(AK60_39, 40, channel=channel) as fake:
        def overheat():
            while not fake.enabled:
                time.sleep(0.01)
            time.sleep(0.5)
            fake.temperature = 75
        threading.Thread(target=overheat, daemon=True).start()
        assert bringup.main(_argv(tmp_path, channel, "AK60-39", "--yes")) == 2
    _, meta = _meta(tmp_path, "AK60-39")
    assert "temperature" in meta["result"]["fault"] and meta["result"]["disable_acknowledged"] is True
