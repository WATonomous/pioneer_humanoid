"""pytest loads this before the tests: lets them `from tum import ...` like the prototypes do."""
import sys
from pathlib import Path

# tum.py lives in terrain_core/terrain_tools/ (parents[1] = the outer terrain_core/ folder)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "terrain_tools"))
