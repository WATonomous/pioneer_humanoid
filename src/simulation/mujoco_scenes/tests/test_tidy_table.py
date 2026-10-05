"""tidy_table: randomisation stays valid and the task checks (steps, dropped, toppled, early) fire when they should.

    pip install mujoco pytest && pip install -e src/pioneer_humanoid -e src/simulation/mujoco_scenes
    pytest src/simulation/mujoco_scenes/tests/test_tidy_table.py
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from humanoid_mujoco_scenes import make_model, scene_progress, scene_reset, scene_step
from humanoid_mujoco_scenes.tidy_table import scene as S
from pioneer_humanoid.mujoco_bimanual_arm import set_home


@pytest.fixture(scope="module")
def model():
    return make_model("tidy_table")


def _episode(model, seed):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    set_home(model, data)
    scene_reset("tidy_table")(model, data, np.random.default_rng(seed))
    mujoco.mj_forward(model, data)
    return data


def _run(model, data, seconds):
    hook = scene_step("tidy_table")
    for _ in range(int(seconds / model.opt.timestep)):
        hook(model, data)
        mujoco.mj_step(model, data)


def _put_in_bin(model, data, obj, slot):
    """Teleport `obj` (an episode_objects entry) upright into its bin; two of a shape sit side by side."""
    bx, by = obj["bin"]
    name = obj["name"][len("obj_"):]
    z = S._stand_height(obj["shape"], obj["size"]) + S.BIN_T + 0.002
    S._place(model, data, name, (bx, by + (0.026 if slot else -0.026), z), 0.0)


def test_randomisation(model):
    for seed in range(20):
        data = _episode(model, seed)
        objs = S.episode_objects(model, data)
        assert S.N_OBJECTS[0] - 1 <= len(objs) <= S.N_OBJECTS[1]
        assert len({o["colour"] for o in objs}) == len(objs)
        assert all(sum(o["shape"] == s for o in objs) <= S.MAX_PER_SHAPE for s in S.SHAPES)
        xy = [data.xpos[model.body(o["name"]).id][:2] for o in objs]
        radii = [S._footprint(o["shape"], o["size"], o["lying"]) for o in objs]
        (x0, x1), (y0, y1) = S.ZONE
        for i, (x, y) in enumerate(xy):
            assert x0 <= x <= x1 and y0 <= y <= y1
            assert S._clear_of_bins(x, y, radii[i])
            for j in range(i):
                assert np.linalg.norm(xy[i] - xy[j]) >= radii[i] + radii[j] + S.FINGER_GAP - 1e-9
        active = {o["name"] for o in objs}
        for name in S._POOL:
            g = model.geom(f"obj_{name}").id
            assert model.geom_contype[g] == (f"obj_{name}" in active)


def test_objects_settle_untouched(model):
    for seed in range(10):
        data = _episode(model, seed)
        start = {o["name"]: data.xpos[model.body(o["name"]).id].copy() for o in S.episode_objects(model, data)}
        _run(model, data, 1.0)
        for name, p in start.items():
            assert np.linalg.norm(data.xpos[model.body(name).id] - p) < 0.003, name
        status = S.episode_status(model, data)
        assert status["step"] == 0 and not (status["dropped"] or status["toppled"] or status["early"])


def test_in_order_succeeds(model):
    data = _episode(model, 3)
    objs = S.episode_objects(model, data)
    used = {s: 0 for s in S.SHAPES}
    for k, obj in enumerate(objs):
        assert scene_progress("tidy_table")(model, data)[0] == k
        _put_in_bin(model, data, obj, used[obj["shape"]])
        used[obj["shape"]] += 1
        _run(model, data, 0.6)
    status = S.episode_status(model, data)
    assert status["step"] == len(objs) and status["tidiness"] == 1.0 and status["success"], status


def test_out_of_order_is_flagged(model):
    data = _episode(model, 3)
    objs = S.episode_objects(model, data)
    _put_in_bin(model, data, objs[1], 0)
    _run(model, data, 0.6)
    status = S.episode_status(model, data)
    assert status["step"] == 0
    assert status["early"] == [f"{objs[1]['colour']} {objs[1]['shape']}"]


def test_wrong_bin_does_not_count(model):
    data = _episode(model, 3)
    obj = S.episode_objects(model, data)[0]
    other = next(s for s in S.SHAPES if s != obj["shape"])
    _put_in_bin(model, data, dict(obj, bin=S.BINS[other]), 0)
    _run(model, data, 0.6)
    status = S.episode_status(model, data)
    assert status["step"] == 0 and status["homed"] == 0


def test_dropped_after_pickup(model):
    data = _episode(model, 3)
    obj = S.episode_objects(model, data)[0]
    p = S._POOL.index(obj["name"][len("obj_"):])
    data.userdata[S._U_FLAGS + p] = S._LIFTED            # as if it had been picked up...
    pos = data.xpos[model.body(obj["name"]).id]
    S._place(model, data, obj["name"][len("obj_"):], (pos[0], pos[1], pos[2] + 0.05), 0.0)   # ...and let go
    _run(model, data, 1.0)
    assert S.episode_status(model, data)["dropped"] == [f"{obj['colour']} {obj['shape']}"]


def test_toppled(model):
    """A box knocked onto its side is toppled; a cylinder that was set out lying down is not."""
    seed, data = next((s, d) for s in range(50) for d in [_episode(model, s)]
                      if {o["shape"] for o in S.episode_objects(model, d)} >= {"box"}
                      and any(o["lying"] for o in S.episode_objects(model, d)))
    box = next(o for o in S.episode_objects(model, data) if o["shape"] == "box")
    pos = data.xpos[model.body(box["name"]).id].copy()
    adr = model.joint(box["name"]).qposadr[0]
    lie = max(box["size"][0], box["size"][1])   # rests on its longest horizontal half-extent... at most
    data.qpos[adr:adr + 7] = [pos[0], pos[1], S.T + lie + 0.002, math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]
    _run(model, data, 1.0)
    assert S.episode_status(model, data)["toppled"] == [f"{box['colour']} box"], seed


def test_hovering_object_is_neither_dropped_nor_home(model):
    """A held object whose finger contact flickers for one step is still and touching nothing: no verdict."""
    data = _episode(model, 3)
    obj = S.episode_objects(model, data)[0]
    name = obj["name"][len("obj_"):]
    p = S._POOL.index(name)
    data.userdata[S._U_FLAGS + p] = S._LIFTED
    bx, by = obj["bin"]
    for xyz in [(0.25, 0.28, S.T + 0.15), (bx, by, S.T + S.BIN_T + S.BIN_WALL_H + 0.02)]:
        S._place(model, data, name, xyz, 0.0)            # placed with zero velocity: "resting" in mid-air
        mujoco.mj_forward(model, data)
        scene_step("tidy_table")(model, data)
        status = S.episode_status(model, data)
        assert not status["dropped"] and status["step"] == 0 and status["homed"] == 0, (xyz, status)


def test_messy_layouts_have_lying_cylinders_and_any_yaw(model):
    lying, yaws = 0, []
    for seed in range(40):
        data = _episode(model, seed)
        for o in S.episode_objects(model, data):
            lying += o["lying"]
            if o["shape"] == "box":
                R = data.xmat[model.body(o["name"]).id].reshape(3, 3)
                yaws.append(math.atan2(R[1, 0], R[0, 0]))
    assert lying > 0
    assert max(np.abs(yaws)) > math.radians(90)
