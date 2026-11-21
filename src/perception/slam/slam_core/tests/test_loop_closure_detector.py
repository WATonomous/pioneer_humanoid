"""Retrieval, and the whole detector on the synthetic world (see slam_core/loop_closure/sim.py)."""
import numpy as np
import pytest

from slam_core.loop_closure import DescriptorDatabase, LoopClosureDetector, LoopParams, Vlad, sim


def test_vlad_ranks_the_same_scene_above_a_different_one():
    rng = np.random.default_rng(0)
    scenes = [rng.normal(size=(40, 64)) for _ in range(3)]     # each scene has its own set of 40 descriptor "prototypes"

    def view(s):                        # 30 of a scene's descriptors, a little noisy
        return scenes[s][rng.choice(40, 30, replace=False)] + rng.normal(0, 0.15, (30, 64))

    vlad = Vlad(k=16)
    vlad.fit([view(s) for s in range(3) for _ in range(5)])
    db = DescriptorDatabase()
    for s in range(3):
        db.add(s, vlad.encode(view(s)))
    for s in range(3):
        ranked = db.query(vlad.encode(view(s)), k=3)
        assert ranked[0][0] == s


def test_bag_of_words_ranks_the_same_scene_above_a_different_one():
    from slam_core.loop_closure import BagOfWords
    rng = np.random.default_rng(1)
    scenes = [rng.normal(size=(40, 64)) for _ in range(3)]

    def view(s):                        # 30 of a scene's descriptors, a little noisy
        return scenes[s][rng.choice(40, 30, replace=False)] + rng.normal(0, 0.15, (30, 64))

    bow = BagOfWords(k=60)
    bow.fit([view(s) for s in range(3) for _ in range(5)])
    db = DescriptorDatabase()
    for s in range(3):
        db.add(s, bow.encode(view(s)))
    for s in range(3):
        assert db.query(bow.encode(view(s)), k=3)[0][0] == s


def test_database_excludes_and_applies_the_score_floor():
    db = DescriptorDatabase()
    for i, v in enumerate([[1, 0], [0.9, 0.1], [0, 1]]):
        db.add(i, v)
    assert [i for i, _ in db.query([1, 0], k=3)] == [0, 1, 2]
    assert [i for i, _ in db.query([1, 0], k=3, exclude=lambda i: i == 0)] == [1, 2]
    assert [i for i, _ in db.query([1, 0], k=3, min_score=0.5)] == [0, 1]


@pytest.fixture(scope="module")
def scenario():
    return sim.make_scenario(seed=0, bias=0.003, rot_sigma=0.02)


def detect(sc, **kw):
    det = LoopClosureDetector(sim.make_verifier(sc), LoopParams(min_inliers=10, **kw))
    return det, sim.run(sc, det, optimize=False)


def test_full_detector_finds_loops_and_none_are_false(scenario):
    det, r = detect(scenario)
    assert r["n_false"] == 0
    assert r["n_long"] >= 10          # revisits after a whole lap, where drift has built up


def test_appearance_only_is_fooled_by_the_identical_hallways(scenario):
    # Same retrieval and verification, but none of the false-loop defenses: the twin hallways get accepted
    det, r = detect(scenario, gate=False, confirm_count=1, pcm=False)
    assert r["n_false"] >= 1


def test_each_defense_rejects_something(scenario):
    det, _ = detect(scenario)
    reasons = {x.reason for x in det.rejected}
    assert {"gate", "unconfirmed"} <= reasons


def test_loops_improve_the_trajectory(scenario):
    r = sim.run(scenario, LoopClosureDetector(sim.make_verifier(scenario), LoopParams(min_inliers=10)))
    assert r["ate_opt"] < 0.2 * r["ate_raw"]


def test_a_loop_can_be_retracted_when_a_better_group_disagrees(scenario):
    # With only the consistency check on, a false loop confirmed early can be taken back out once true loops
    # arrive. Whatever remains active must be mutually consistent and contain no false loops
    det, r = detect(scenario, gate=False, confirm_count=1, pcm=True)
    assert r["n_false"] == 0


def test_vlad_retrieval_end_to_end(scenario):
    # Local descriptors -> VLAD vocabulary -> global vectors -> retrieval -> the full detector, on the simulator
    import dataclasses
    loc = sim.local_descriptors(scenario)
    vlad = Vlad(32)
    vlad.fit(loc[:150] + loc[300:350])
    sc = dataclasses.replace(scenario, descs=[vlad.encode(d) for d in loc])
    det = LoopClosureDetector(sim.make_verifier(sc), LoopParams(min_inliers=10, min_score=0.3))
    r = sim.run(sc, det, optimize=False)
    assert r["n_false"] == 0 and r["n_long"] >= 10
