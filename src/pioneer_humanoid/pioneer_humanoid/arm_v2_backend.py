"""Load the repo-owned Arm V2 mechanism without importing Isaac Lab."""
from functools import lru_cache
import importlib.util
from pathlib import Path

ASSET_ROOT = Path(__file__).resolve().parents[3]/'assets'/'pioneer_bimanual_arm'


@lru_cache(maxsize=1)
def mechanism():
    path = ASSET_ROOT/'simulation'/'articulated_arm.py'
    spec = importlib.util.spec_from_file_location('_pioneer_arm_v2_mechanism', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def initialize_gripper_linkages(model, data, prefix=''):
    """Home reset uses zero jaw displacement, crank angle and passive pin angles."""
    for side in ['right', 'left']:
        for name in [f'{side}_gripper', f'{side}_linkage_pin0', f'{side}_linkage_pin1']:
            data.qpos[model.joint(prefix+name).qposadr[0]] = 0.
