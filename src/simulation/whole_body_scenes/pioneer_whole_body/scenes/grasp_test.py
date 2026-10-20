"""Grasp test bench: a knob bolt standing in a rack, a fixed plate with one threaded hole."""
import mujoco

from pioneer_whole_body import scene
from pioneer_whole_body.parts import STEEL, Thread, add_bolt_weld, add_knob_bolt, add_thread_hole
from pioneer_whole_body.studio import BENCH_TOP_Z, add_bench, add_studio

T = BENCH_TOP_Z
BOLT_POS = (0.50, -0.17)
HOLE_POS = (0.50, -0.03)
PLATE_H = 0.02
TRAVEL = 0.009
THREADS = [Thread("hole", ("bolt",), TRAVEL)]


def add_rack(world, name, pos, n=1, pitch=0.06):
    rack = world.add_body(name=name, pos=list(pos))
    blue = [0.2, 0.35, 0.6, 1]
    for i in range(n):
        y = i * pitch
        for dx, dy, sx, sy in ((0.009, 0, 0.004, 0.013), (-0.009, 0, 0.004, 0.013),
                               (0, 0.009, 0.005, 0.004), (0, -0.009, 0.005, 0.004)):
            rack.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, 0.012], pos=[dx, y + dy, 0.012], rgba=blue)
    return rack


def _step(model, data):
    if not hasattr(model, "_threads"):
        pass
    for th in _bound(model):
        th.step(data)
        th.try_engage(model, data, set())


_BOUND = {}


def _bound(model):
    key = id(model)
    if key not in _BOUND:
        _BOUND[key] = [Thread(t.hole, t.bolts, t.travel, t.radius).bind(model) for t in THREADS]
    return _BOUND[key]


@scene("grasp_test", step=_step, robot_pos=(-0.3, 0, 0))
def build(spec):
    add_studio(spec)
    add_bench(spec)
    world = spec.worldbody
    add_rack(world, "rack", [BOLT_POS[0], BOLT_POS[1], T])
    add_knob_bolt(world, "bolt", [BOLT_POS[0], BOLT_POS[1], T + 0.0005])
    plate = world.add_body(name="plate", pos=[HOLE_POS[0], HOLE_POS[1], T])
    plate.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, PLATE_H / 2], pos=[0, 0, PLATE_H / 2], rgba=STEEL)
    add_thread_hole(spec, plate, "hole", [0, 0, PLATE_H], TRAVEL)
    add_bolt_weld(spec, "hole", "bolt")
    spec.add_exclude(bodyname1="plate", bodyname2="bolt")
