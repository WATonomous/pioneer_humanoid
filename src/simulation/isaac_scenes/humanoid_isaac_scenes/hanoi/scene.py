"""Tower of Hanoi in the lightbox workcell: wooden base, pegs A B C, three rings stacked on A, hand-over tray T.

Each arm points straight down only on its own side (left y > ~+0.03, right y < ~-0.03), so a ring going between
A and C is passed through the tray, which both arms reach. Props: ``assets/props/hanoi`` (``make_props.py``).
"""
from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from humanoid_isaac_scenes import scene
from humanoid_rl_tasks.workcell import BOTTOM_PANEL, ROBOT_BASE_POS, TABLE_TOP_Z, LightboxWorkcellCfg

_PROPS = Path(__file__).resolve().parents[5] / "assets" / "props" / "hanoi"

TOP = TABLE_TOP_Z + BOTTOM_PANEL               # the white board the base stands on
PEGS = {"A": (0.51, 0.18), "B": (0.51, 0.0), "C": (0.51, -0.18)}
TRAY = (0.41, 0.0)
BOARD_H = 0.02                                 # base and tray height
BOARD_TOP = TOP + BOARD_H
DISK_H, HOLE_R, PEG_R = 0.022, 0.011, 0.005
PEG_H = 3 * DISK_H + 0.018
DISKS = {"orange": 0.064, "yellow": 0.054, "green": 0.044}   # diameter, bottom to top


@scene("hanoi", robot_pos=ROBOT_BASE_POS, camera=([0.97, -0.22, 1.22], [0.36, 0.0, 0.90]))
@configclass
class HanoiSceneCfg(LightboxWorkcellCfg):
    base_board = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/HanoiBase", init_state=AssetBaseCfg.InitialStateCfg(pos=(0.51, 0.0, TOP + BOARD_H / 2)),
        spawn=sim_utils.CuboidCfg(size=(0.10, 0.48, BOARD_H), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.40, 0.22), roughness=0.5)))
    tray = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Tray", init_state=AssetBaseCfg.InitialStateCfg(pos=(*TRAY, TOP)),
                        spawn=sim_utils.UsdFileCfg(usd_path=str(_PROPS / "tray.usda")))


for _k, _xy in PEGS.items():
    setattr(HanoiSceneCfg, f"peg_{_k}", AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Peg_{_k}", init_state=AssetBaseCfg.InitialStateCfg(pos=(*_xy, BOARD_TOP)),
        spawn=sim_utils.UsdFileCfg(usd_path=str(_PROPS / "peg.usda"))))
for _i, _n in enumerate(DISKS):   # ring origin: bottom centre
    setattr(HanoiSceneCfg, f"disk_{_n}", RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Disk_{_n}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*PEGS["A"], BOARD_TOP + DISK_H * _i + 0.001 * (_i + 1))),
        spawn=sim_utils.UsdFileCfg(usd_path=str(_PROPS / f"ring_{_n}.usda"),
                                   rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.5),
                                   collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0))))
