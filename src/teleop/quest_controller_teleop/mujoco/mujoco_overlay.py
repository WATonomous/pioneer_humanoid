"""What the controller is asking for, drawn in the MuJoCo viewer: the reset key, as small top-left text."""
from __future__ import annotations


def show_text(viewer, text: str) -> None:
    import mujoco

    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_50, mujoco.mjtGridPos.mjGRID_TOPLEFT, text, ""))
