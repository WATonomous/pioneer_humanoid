"""Synthetic takes through sync -> quality -> dataset vectors, against the known motion."""
import numpy as np
import pytest

from synthetic import make_take, synthetic_config, true_closure, true_grip_pose
from umi_gripper import config as umi_config
from umi_gripper import geometry
from umi_gripper.postprocess.quality import check_episode, trim_invalid_ends
from umi_gripper.postprocess.sync import episode_vectors, nearest_frames, sync_take
from umi_gripper.postprocess.to_lerobot import evaluate_takes, load_schema
from umi_gripper.take import find_takes, load_take, next_take_path


def test_take_round_trip(tmp_path):
    path = make_take(tmp_path / "s" / "take_0000", duration=1.0)
    take = load_take(path)
    assert take.meta["task"] == "synthetic"
    assert take.poses["position"].shape[1] == 3 and len(take.camera_s) == 30
    assert take.load_frame(3).shape == (480, 640, 3)
    assert find_takes(tmp_path) == [path]
    assert next_take_path(tmp_path / "s").name == "take_0001"


def test_sync_recovers_the_tcp_motion(tmp_path):
    cfg = synthetic_config()
    episode = sync_take(load_take(make_take(tmp_path / "t", duration=4.0)), cfg)
    assert episode.valid.all() and not episode.issues
    assert np.allclose(np.diff(episode.times), 1 / 25)
    grip_to_tcp = umi_config.grip_to_tcp(cfg)
    for i in range(0, len(episode), 10):
        # the clock mapping keeps the 3 ms minimum delay of the synthetic transport; < 5 mm at hand speeds
        expected = true_grip_pose(episode.times[i]) @ grip_to_tcp
        assert np.linalg.norm(episode.tcp[i, :3, 3] - expected[:3, 3]) < 0.005
        assert episode.closure[i] == pytest.approx(true_closure(episode.times[i]), abs=0.01)
    assert (episode.frame_index >= 0).all()


def test_camera_latency_is_removed(tmp_path):
    cfg = synthetic_config(latency={"camera_s": 0.05})
    episode = sync_take(load_take(make_take(tmp_path / "t", camera_latency_s=0.05)), cfg)
    take = episode.take
    frame_t = take.camera_s[episode.frame_index] - 0.05
    assert np.max(np.abs(frame_t - episode.times)) <= 1 / 60 + 1e-9


def test_interior_tracking_gap_rejects_and_end_gaps_are_trimmed(tmp_path):
    cfg = synthetic_config()
    gap = check_episode(sync_take(load_take(make_take(tmp_path / "gap", untracked=[(2.0, 2.3)])), cfg),
                        cfg["postprocess"])
    assert not gap.ok and any("lost tracking" in r for r in gap.reasons)
    short_gap = check_episode(sync_take(load_take(make_take(tmp_path / "short", untracked=[(2.0, 2.05)])), cfg),
                              cfg["postprocess"])
    assert short_gap.ok, short_gap.reasons   # 50 ms is under max_tracking_gap_s: interpolated
    ends = check_episode(sync_take(load_take(make_take(tmp_path / "ends", untracked=[(0, 0.5), (3.6, 4)])), cfg),
                         cfg["postprocess"])
    assert ends.ok, ends.reasons
    assert ends.episode.times[0] >= 0.5 and ends.episode.times[-1] <= 3.6


def test_jump_and_recentre_and_short_takes_reject(tmp_path):
    cfg = synthetic_config()
    post = cfg["postprocess"]
    jump = check_episode(sync_take(load_take(make_take(tmp_path / "jump", jump_at=2.0)), cfg), post)
    assert not jump.ok and any("jump" in r for r in jump.reasons)
    recentre = check_episode(sync_take(load_take(make_take(tmp_path / "re", epochs=lambda t: int(t > 2))), cfg), post)
    assert not recentre.ok and any("re-centred" in r for r in recentre.reasons)
    short = check_episode(sync_take(load_take(make_take(tmp_path / "short", duration=0.6)), cfg), post)
    assert not short.ok and any("too short" in r for r in short.reasons)


def test_no_camera_take(tmp_path):
    cfg = synthetic_config()
    episode = sync_take(load_take(make_take(tmp_path / "nc", camera=False)), cfg)
    assert episode.frame_index is None and episode.valid.all()


def test_episode_vectors_are_start_relative_and_actions_lead(tmp_path):
    cfg = synthetic_config()
    episode = sync_take(load_take(make_take(tmp_path / "t")), cfg)
    state, action = episode_vectors(episode)
    assert state.shape == action.shape == (len(episode), 10) and state.dtype == np.float32
    np.testing.assert_allclose(state[0, :9], [0, 0, 0, 1, 0, 0, 0, 1, 0], atol=1e-6)
    np.testing.assert_array_equal(action[:-1], state[1:])
    np.testing.assert_array_equal(action[-1], state[-1])
    # relative motion between two steps is the same in the world and in the start frame
    a, b = 5, 40
    world = geometry.invert(episode.tcp[a]) @ episode.tcp[b]
    local = geometry.invert(geometry.vector_to_pose(state[a, :9])) @ geometry.vector_to_pose(state[b, :9])
    np.testing.assert_allclose(local, world, atol=1e-5)
    # invariant to where the headset put its world frame
    shifted = episode.tcp.copy()
    shift = geometry.make_pose(geometry.rotation_exp([0, 1.0, 0]), [3, 0, 1])
    episode.tcp = np.einsum("ij,njk->nik", shift, shifted)
    state2, _ = episode_vectors(episode)
    np.testing.assert_allclose(state2, state, atol=1e-5)


def test_nearest_frames():
    frames = np.array([0.0, 0.033, 0.066])
    np.testing.assert_array_equal(nearest_frames(frames, np.array([-0.001, 0.02, 0.07, 0.2]), 0.02), [0, 1, 2, -1])
    np.testing.assert_array_equal(nearest_frames(np.array([]), np.array([0.0]), 0.02), [-1])


def test_trim_all_invalid():
    from umi_gripper.postprocess.sync import Episode
    ep = Episode(None, np.zeros(3), np.tile(np.eye(4), (3, 1, 1)), np.zeros(3), None, np.zeros(3, bool))
    assert len(trim_invalid_ends(ep)) == 0


def test_evaluate_takes_report_and_schema(tmp_path):
    cfg = synthetic_config()
    good = make_take(tmp_path / "s" / "take_0000")
    bad = make_take(tmp_path / "s" / "take_0001", untracked=[(1.0, 2.0)])
    nocam = make_take(tmp_path / "s" / "take_0002", camera=False)
    entries = evaluate_takes(find_takes(tmp_path), cfg, require_camera=True)
    assert [e["ok"] for e in entries] == [True, False, False]
    assert entries[0]["task"] == "synthetic" and entries[0]["duration_s"] > 3
    assert any("no camera" in r for r in entries[2]["reasons"])
    assert {str(good), str(bad), str(nocam)} == {e["take"] for e in entries}
    from umi_gripper._paths import ROBOT_LEARNING, add
    add(ROBOT_LEARNING)
    from humanoid_robot_learning.recording.schema import build_features
    features = build_features(load_schema())
    assert features["observation.state"]["shape"] == (10,) and features["action"]["shape"] == (10,)
    assert features["observation.images.wrist"]["shape"] == (480, 640, 3)
    assert int(load_schema()["fps"]) == int(cfg["postprocess"]["fps"])
    assert "observation.images.wrist" not in build_features(load_schema(no_images=True))


def test_to_lerobot_dry_run_cli(tmp_path, capsys):
    from umi_gripper.postprocess import to_lerobot
    make_take(tmp_path / "s" / "take_0000")
    assert to_lerobot.main([str(tmp_path), "--dry-run", "--calibration", str(tmp_path / "none.json")]) == 0
    assert "1/1 takes pass" in capsys.readouterr().out
