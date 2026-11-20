import numpy as np
import pytest

from umi_gripper.clock import headset_to_host, sliding_min
from umi_gripper.quest_tracker import PacketHandler
from umi_gripper.width import EncoderWidth, TriggerWidth, check_width_calibration, closure_from_counts


def test_sliding_min_matches_brute_force():
    rng = np.random.default_rng(0)
    t = np.sort(rng.random(300) * 10)
    v = rng.normal(size=300)
    out = sliding_min(t, v, 0.7)
    brute = [v[np.abs(t - ti) <= 0.7].min() for ti in t]
    np.testing.assert_allclose(out, brute)


def test_headset_to_host_removes_jitter_keeps_min_delay():
    rng = np.random.default_rng(1)
    true = np.arange(0, 20, 1 / 72)
    client_ms = (true - 500.0) * 1000
    host = true + 0.003 + rng.exponential(0.004, size=len(true))   # 3 ms minimum delay + jitter
    mapped = headset_to_host(client_ms, host)
    assert np.max(np.abs(mapped - (true + 0.003))) < 0.002
    with pytest.raises(ValueError):
        headset_to_host(client_ms[::-1], host)


def test_closure_from_counts_including_wrap():
    assert closure_from_counts(1000, 1000, 1400) == 0.0
    assert closure_from_counts(1200, 1000, 1400) == pytest.approx(0.5)
    assert closure_from_counts(1500, 1000, 1400) == 1.0          # clamped
    assert closure_from_counts(4000, 3900, 100) == pytest.approx(100 / 296)   # through 4095 -> 0
    assert closure_from_counts(1200, 1400, 1000) == pytest.approx(0.5)        # closing decreases the count


def test_width_calibration_checks():
    with pytest.raises(ValueError, match="not calibrated"):
        check_width_calibration({"open_count": None, "closed_count": 10})
    with pytest.raises(ValueError, match="apart"):
        check_width_calibration({"open_count": 100, "closed_count": 120})
    assert check_width_calibration({"open_count": 100, "closed_count": 900}) == (100, 900)


class FakeEncoder:
    def __init__(self):
        self.count = 1000
        self.closed = False

    def read_count(self):
        return self.count

    def close(self):
        self.closed = True


def test_encoder_width_thread_samples_and_closes():
    import time
    encoder = FakeEncoder()
    width = EncoderWidth({"open_count": 1000, "closed_count": 1400, "rate_hz": 200}, encoder=encoder).start()
    time.sleep(0.05)
    encoder.count = 1300
    time.sleep(0.05)
    width.stop()
    samples = width.drain()
    assert len(samples) > 5 and encoder.closed and width.error is None
    assert samples[0][1] == 0.0 and samples[-1][1] == pytest.approx(0.75)
    assert all(b[0] >= a[0] for a, b in zip(samples, samples[1:]))


def packet(sequence, side="right", tracked=True, buttons=None, head=True, trigger=0.3):
    controller = {"tracked": True, "position": [0.1, 1.0, -0.3], "orientation": [0, 0, 0, 1], "trigger": trigger,
                  **(buttons or {})} if tracked else {"tracked": False}
    return {"version": 1, "head": [0, 0, 0, 1] if head else None, "left": None, "right": None, side: controller,
            "sample": {"sequence": sequence, "time_ms": 10.0 * sequence, "epoch": 0}}


def test_packet_handler_queues_every_sample_and_drops_out_of_order():
    handler = PacketHandler("right")
    for seq in (1, 2, 3):
        assert handler.handle(packet(seq), host_s=seq * 0.01)
    assert not handler.handle(packet(2), host_s=0.04)
    handler.handle(packet(4, tracked=False), host_s=0.05)
    samples = handler.drain()
    assert [s.client_ms for s in samples] == [10, 20, 30, 40]
    assert [s.tracked for s in samples] == [True, True, True, False]
    np.testing.assert_allclose(samples[0].position, [0.1, 1.0, -0.3])
    assert samples[0].trigger == pytest.approx(0.3)
    assert handler.drain() == []


def test_untracked_without_head_pose():
    handler = PacketHandler("right")
    handler.handle(packet(1, head=False), 0.0)
    assert not handler.drain()[0].tracked


def test_button_edges_on_the_mounted_controller():
    right = PacketHandler("right")
    states = [{}, {"record": True}, {"record": True}, {}, {"precision": True}, {"calibrate": True}]
    for seq, buttons in enumerate(states, 1):
        right.handle(packet(seq, buttons=buttons), seq)
    assert [e.name for e in right.poll_events()] == ["primary", "secondary", "stick"]
    left = PacketHandler("left")
    left.handle(packet(1, side="left", buttons={"reset_scene": True}), 1)   # Y on the left controller
    left.handle(packet(2, side="left", buttons={"record": True}), 2)        # not a left-controller button
    assert [e.name for e in left.poll_events()] == ["primary"]


def test_malformed_packet_raises():
    handler = PacketHandler("right")
    bad = packet(1)
    bad["right"]["position"] = [float("nan"), 0, 0]
    with pytest.raises(ValueError):
        handler.handle(bad, 0.0)


def test_trigger_width_uses_tracked_samples_only():
    handler = PacketHandler("right")
    handler.handle(packet(1, trigger=0.2), 0.0)
    handler.handle(packet(2, tracked=False), 0.01)
    width = TriggerWidth()
    width.feed(handler.drain())
    assert width.drain() == [(0.0, 0.2, 0.2)]
