"""Synthetic takes through sync -> quality -> dataset vectors, against the known motion."""
import numpy as np
import pytest

from synthetic import make_take, synthetic_config, true_closure, true_grip_pose
from umi_gripper import config as umi_config
from umi_gripper import geometry
from umi_gripper.postprocess.quality import runs, split_episode
from umi_gripper.postprocess.sync import episode_extras, episode_vectors, nearest_frames, sync_take
from umi_gripper.postprocess.to_lerobot import evaluate_takes, load_schema
from umi_gripper.take import find_takes, load_take, next_take_path


def test_take_round_trip(tmp_path):
    path = make_take(tmp_path / "s" / "take_0000", duration=1.0)
    take = load_take(path)
    assert take.meta["task"] == "synthetic"
    assert take.poses["position"].shape[1] == 3 and len(take.camera_s) == 25
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
    assert (episode.frame_index >= 0).all() and len(np.unique(episode.frame_index)) == len(episode)


def test_steps_are_at_their_frames_capture_time(tmp_path):
    cfg = synthetic_config(latency={"camera_s": 0.05})
    episode = sync_take(load_take(make_take(tmp_path / "t", camera_latency_s=0.05)), cfg)
    frame_t = episode.take.camera_s[episode.frame_index] - 0.05
    np.testing.assert_allclose(episode.times, frame_t, atol=1e-12)


def test_camera_rate_that_does_not_divide_the_dataset_rate_is_reported(tmp_path):
    episode = sync_take(load_take(make_take(tmp_path / "t", camera_hz=30.0)), synthetic_config())
    assert any("30.0 fps" in issue for issue in episode.issues)
    assert not sync_take(load_take(make_take(tmp_path / "u", camera_hz=50.0)), synthetic_config()).issues


def test_dropped_frame_invalidates_its_step(tmp_path):
    cfg = synthetic_config()
    path = make_take(tmp_path / "t")
    with np.load(path / "camera.npz") as data:
        stamps = data["host_s"]
    np.savez(path / "camera.npz", host_s=np.delete(stamps, np.arange(45, 48)))   # 3 frames lost (0.12 s)
    episode = sync_take(load_take(path), cfg)
    assert 0 < (~episode.valid).sum() <= 3
    assert len(np.unique(episode.frame_index[episode.valid])) == episode.valid.sum()


def test_tracking_gap_splits_and_short_gap_is_interpolated(tmp_path):
    cfg = synthetic_config()
    post = cfg["postprocess"]
    split = split_episode(sync_take(load_take(make_take(tmp_path / "gap", duration=6.0, untracked=[(2.5, 2.8)])), cfg),
                          post)
    assert split.ok and len(split.episodes) == 2, split.reasons
    assert split.episodes[0].times[-1] < 2.5 and split.episodes[1].times[0] > 2.8
    assert any("lost tracking" in r for r in split.reasons)
    short_gap = split_episode(sync_take(load_take(make_take(tmp_path / "short", untracked=[(2.0, 2.05)])), cfg), post)
    assert len(short_gap.episodes) == 1 and short_gap.stats["kept_fraction"] == 1.0
    ends = split_episode(sync_take(load_take(make_take(tmp_path / "ends", untracked=[(0, 0.5), (3.6, 4)])), cfg), post)
    assert len(ends.episodes) == 1
    assert ends.episodes[0].times[0] >= 0.5 and ends.episodes[0].times[-1] <= 3.6


def test_jump_and_recentre_split(tmp_path):
    cfg = synthetic_config()
    post = cfg["postprocess"]
    jump = split_episode(sync_take(load_take(make_take(tmp_path / "jump", duration=5.0, jump_at=2.5)), cfg), post)
    assert len(jump.episodes) == 2 and any("1 TCP jumps" in r for r in jump.reasons)
    for episode in jump.episodes:   # each piece is smooth on its own
        assert np.linalg.norm(np.diff(episode.tcp[:, :3, 3], axis=0), axis=1).max() < 0.05
    recentred = make_take(tmp_path / "re", duration=5.0, epochs=lambda t: int(t > 2.5))
    recentre = split_episode(sync_take(load_take(recentred), cfg), post)
    assert len(recentre.episodes) == 2 and recentre.stats["segment_changes"] == 1


def test_short_take_gives_nothing(tmp_path):
    cfg = synthetic_config()
    split = split_episode(sync_take(load_take(make_take(tmp_path / "short", duration=0.6)), cfg), cfg["postprocess"])
    assert not split.ok and any("shorter" in r for r in split.reasons)


def test_no_camera_take(tmp_path):
    cfg = synthetic_config()
    episode = sync_take(load_take(make_take(tmp_path / "nc", camera=False)), cfg)
    assert episode.frame_index is None and episode.valid.all()
    assert np.allclose(np.diff(episode.times), 1 / 25)


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
    shift = geometry.make_pose(geometry.rotation_exp([0, 1.0, 0]), [3, 0, 1])
    episode.tcp = np.einsum("ij,njk->nik", shift, episode.tcp)
    state2, _ = episode_vectors(episode)
    np.testing.assert_allclose(state2, state, atol=1e-5)


def test_episode_extras_absolute_columns(tmp_path):
    cfg = synthetic_config()
    episode = sync_take(load_take(make_take(tmp_path / "t")), cfg)
    extras = episode_extras(episode, cfg)
    world = extras["observation.eef_pose_world"]
    assert world.shape == (len(episode), 9)
    np.testing.assert_allclose(geometry.vector_to_pose(world[7]), episode.tcp[7], atol=1e-5)
    np.testing.assert_array_equal(extras["observation.demo_start_pose_world"], np.repeat(world[:1], len(episode), 0))
    width = extras["observation.gripper_width_m"][:, 0]
    np.testing.assert_allclose(width, (1 - episode.closure) * cfg["width"]["max_width_m"], atol=1e-6)


def test_nearest_frames():
    frames = np.array([0.0, 0.033, 0.066])
    np.testing.assert_array_equal(nearest_frames(frames, np.array([-0.001, 0.02, 0.07, 0.2]), 0.02), [0, 1, 2, -1])
    np.testing.assert_array_equal(nearest_frames(np.array([]), np.array([0.0]), 0.02), [-1])


def test_runs():
    assert runs(np.array([1, 1, 0, 1, 0, 0, 1], bool)) == [(0, 2), (3, 4), (6, 7)]
    assert runs(np.zeros(3, bool)) == []


def test_evaluate_takes_report_and_schema(tmp_path):
    cfg = synthetic_config()
    make_take(tmp_path / "s" / "take_0000")
    make_take(tmp_path / "s" / "take_0001", duration=6.0, untracked=[(2.5, 2.8)])
    make_take(tmp_path / "s" / "take_0002", untracked=[(0.0, 4.0)])
    make_take(tmp_path / "s" / "take_0003", camera=False)
    entries = evaluate_takes(find_takes(tmp_path), cfg, require_camera=True)
    assert [e["ok"] for e in entries] == [True, True, False, False]
    assert entries[0]["task"] == "synthetic" and entries[0]["episodes"] == 1 and entries[0]["duration_s"] > 3
    assert entries[1]["episodes"] == 2 and entries[1]["kept_fraction"] < 1
    assert any("no camera" in r for r in entries[3]["reasons"])
    from umi_gripper._paths import ROBOT_LEARNING, add
    add(ROBOT_LEARNING)
    from humanoid_robot_learning.recording.schema import build_features
    features = build_features(load_schema())
    assert features["observation.state"]["shape"] == (10,) and features["action"]["shape"] == (10,)
    assert features["observation.images.wrist"]["shape"] == (480, 640, 3)
    assert features["observation.eef_pose_world"]["shape"] == (9,)
    assert features["observation.demo_start_pose_world"]["shape"] == (9,)
    assert features["observation.gripper_width_m"]["shape"] == (1,)
    assert set(episode_extras(entries[0]["_split"].episodes[0], cfg)) == set(load_schema()["extra_features"])
    assert int(load_schema()["fps"]) == int(cfg["postprocess"]["fps"])
    assert "observation.images.wrist" not in build_features(load_schema(no_images=True))


def test_to_lerobot_dry_run_cli(tmp_path, capsys):
    from umi_gripper.postprocess import to_lerobot
    make_take(tmp_path / "s" / "take_0000")
    assert to_lerobot.main([str(tmp_path), "--dry-run", "--calibration", str(tmp_path / "none.json")]) == 0
    assert "1/1 takes give 1 episodes" in capsys.readouterr().out


def test_to_lerobot_writes_dataset(tmp_path):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from umi_gripper.postprocess import to_lerobot
    make_take(tmp_path / "raw" / "s" / "take_0000", duration=6.0, untracked=[(2.5, 2.8)])
    out = tmp_path / "ds" / "real"
    assert to_lerobot.main([str(tmp_path / "raw"), "--out", str(out), "--calibration", str(tmp_path / "none.json")]) == 0
    dataset = LeRobotDataset("humanoid/pioneer_umi", root=out)
    assert dataset.num_episodes == 2
    item = dataset[3]
    assert item["observation.state"].shape == (10,) and item["observation.eef_pose_world"].shape == (9,)
    assert item["observation.images.wrist"].shape == (3, 480, 640)
    assert (tmp_path / "ds" / "real_umi_report.json").is_file()
