"""CPU checks for the frozen source robot; USD is generated at runtime."""

import hashlib
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[4]
ASSET_ROOT = REPO_ROOT / "assets/old_leg_only_humanoid"
# Captured before upstream's asset rename. These bytes define the trained robot;
# relocation must not silently adopt the newer shared URDF or mesh geometry.
BASELINE_SHA256 = {
    "rl_baseline_v1.urdf": "1846f2965ed9335300e1e4b98977e1145f0ac651decb15978e0394f09d9ece30",
    "meshes/Ankle_L.STL": "c8af275215bb21635aae7abce3ebe4af39d5f4e709dad4ff66ab44dbe26d4005",
    "meshes/Ankle_R.STL": "8aafe9d2f85b10c245bab14bc41de1b7c644a51a30801a11b2f609c35507fd27",
    "meshes/Calf_L.STL": "458ffba7deac46989d822204394550534f1c712fcb51a5f0a3a062ff492f2c59",
    "meshes/Calf_R.STL": "2de0996f188fb63292616603f727f7b52916ff230ba93f72d4d3427ba0e05ea6",
    "meshes/Foot_L.STL": "77094e85b90b62ab604e5e55ea83a4b2d5b1857fd59d92a660893f8bacdc9ae0",
    "meshes/Foot_R.STL": "08d1ea07594c1853e202759720ccfc1079ea6adea49cda59e378e32c12e06d98",
    "meshes/Hip_A_L.STL": "f2ed0ff06f0330cfcb3b33b9f09b3aa25dbf3ecda09210e7d734b17f3c6592cc",
    "meshes/Hip_A_R.STL": "95e1950e2acd7cb97e8e9cf9261a61954eedc4eb0c0432c545afaed42090644b",
    "meshes/Hip_F_L.STL": "819afbfbaad899a200e75fbf324a19502e384c908e459d2b09cc2ef08d3d4710",
    "meshes/Hip_F_R.STL": "ea855e92c05ac0e8d6ae8ede1c42232ed3a7e945202e82af384a7931c002ed91",
    "meshes/Thigh_L.STL": "ee3a50685ed74717fb67f4f3792b8d51776211d81530fc4676995d43b553c517",
    "meshes/Thigh_R.STL": "c9de359ede2eb2db46d8a054439adcffcd6470f924501ab8c9d1fd0a6706c444",
    "meshes/base_link.STL": "3fc700aab6eb476835d7451c557fb8cb34fd9a116be2eebf2cdb797986784aa8",
}


class BaselineAssets(unittest.TestCase):
    def test_source_robot_and_shared_meshes_preserve_baseline_bytes(self):
        for relative, expected in BASELINE_SHA256.items():
            with self.subTest(path=relative):
                self.assertEqual(hashlib.sha256((ASSET_ROOT / relative).read_bytes()).hexdigest(), expected)

    def test_every_referenced_urdf_mesh_is_covered_by_the_baseline_contract(self):
        urdf = ET.parse(ASSET_ROOT / "rl_baseline_v1.urdf")
        meshes = {mesh.attrib["filename"] for mesh in urdf.iter("mesh")}
        self.assertEqual(meshes, {name for name in BASELINE_SHA256 if name.startswith("meshes/")})


if __name__ == "__main__":
    unittest.main()
