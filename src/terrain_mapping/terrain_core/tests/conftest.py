"""pytest loads this before the tests: lets them `from tum import ...` like the prototypes do."""
import sys
from pathlib import Path

# parents[1] = the outer terrain_core/ folder (contains terrain_core/ and terrain_tools/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "terrain_tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "terrain_mapping_bringup" / "scripts"))
