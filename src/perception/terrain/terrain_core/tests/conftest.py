"""pytest loads this before the tests: lets them `from tum import ...` like the prototypes do."""
import sys
from pathlib import Path

# tum.py lives in terrain/terrain_tools/ (parents[2] = the terrain/ folder)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "terrain_tools"))
