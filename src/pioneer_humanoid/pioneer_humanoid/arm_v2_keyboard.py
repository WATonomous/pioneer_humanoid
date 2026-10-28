"""Run with python -m pioneer_humanoid.arm_v2_keyboard (Windows or Linux)."""
import runpy
import sys
from .arm_v2_backend import ASSET_ROOT


def main():
    directory = str(ASSET_ROOT/'simulation')
    sys.path.insert(0, directory)
    try:
        runpy.run_path(str(ASSET_ROOT/'simulation'/'keyboard_control.py'), run_name='__main__')
    finally:
        sys.path.remove(directory)


if __name__ == '__main__':
    main()
