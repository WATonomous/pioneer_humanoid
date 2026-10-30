"""CPU-only contracts for the checked-in Isaac Lab URDF conversion marker.

Isaac Lab compares ``file.readline()`` directly with its bare MD5 digest, so a
newline is meaningful here. The optional PyYAML check recomputes that digest
using the recorded converter settings and URDF bytes; it does not validate the
USD output itself or guarantee converter compatibility across Isaac versions.
"""

import hashlib
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

try:
    import yaml
except ImportError:
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[4]
ASSET_ROOT = REPO_ROOT / "assets/old_leg_only_humanoid"
CACHE_ROOT = ASSET_ROOT / "usd/rl_baseline_v1"
HASH_FILE = CACHE_ROOT / ".asset_hash"
# Captured before upstream's asset rename. These bytes define the trained robot;
# relocation must not silently adopt the newer shared URDF or converted USD.
BASELINE_SHA256 = {
    "rl_baseline_v1.urdf": "1846f2965ed9335300e1e4b98977e1145f0ac651decb15978e0394f09d9ece30",
    "usd/rl_baseline_v1/whole_body_humanoid.usd": "dddc6445093303a41be0da0e42a81ab6bdb87c4f9243a4e976cf79e2ddfab175",
    "usd/rl_baseline_v1/configuration/Wato_Humanoid_V1_base.usd": "413ba1a268439aa1cd8de439865045e81497518465623e7726b6519a9dd8d26f",
    "usd/rl_baseline_v1/configuration/Wato_Humanoid_V1_physics.usd": "6baa7725b2842a1def2a5a818a1608b2d4a2fb13852fba972690eec2e6565ebf",
    "usd/rl_baseline_v1/configuration/Wato_Humanoid_V1_robot.usd": "c2b6898ba9742a3f2074cf182bb5beea9c59ae7e40d01ec5f1d6cb3722d91d4f",
    "usd/rl_baseline_v1/configuration/Wato_Humanoid_V1_sensor.usd": "75b762ed81c363d6636fae7f3601df4daa1ccdde02454654b0d479d5a47af2b1",
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


class AssetCacheContracts(unittest.TestCase):
    def test_versioned_robot_and_all_cached_layers_preserve_baseline_bytes(self):
        for relative, expected in BASELINE_SHA256.items():
            with self.subTest(path=relative):
                self.assertEqual(hashlib.sha256((ASSET_ROOT / relative).read_bytes()).hexdigest(), expected)

    def test_every_referenced_urdf_mesh_is_covered_by_the_baseline_contract(self):
        urdf = ET.parse(ASSET_ROOT / "rl_baseline_v1.urdf")
        meshes = {mesh.attrib["filename"] for mesh in urdf.iter("mesh")}
        self.assertEqual(meshes, {name for name in BASELINE_SHA256 if name.startswith("meshes/")})

    def test_marker_is_bare_md5_without_a_line_ending(self):
        marker = HASH_FILE.read_bytes()
        self.assertRegex(marker, rb"\A[0-9a-f]{32}\Z")
        self.assertEqual(len(marker), 32)
        with HASH_FILE.open(encoding="ascii", newline="") as stream:
            self.assertEqual(stream.readline(), marker.decode("ascii"))
            self.assertEqual(stream.read(), "")

    @unittest.skipIf(yaml is None, "PyYAML is needed to read the recorded converter configuration")
    def test_marker_matches_recorded_converter_settings_and_urdf(self):
        config = yaml.safe_load((CACHE_ROOT / "config.yaml").read_text(encoding="utf-8"))
        # These are precisely the path fields AssetConverterBase._config_to_hash
        # removes. Preserve insertion order and json.dumps defaults: sorting keys
        # or changing separators would not reproduce Isaac Lab's converter hash.
        for name in ("asset_path", "usd_dir", "usd_file_name"):
            config.pop(name)
        digest = hashlib.md5(json.dumps(config).encode())
        with (ASSET_ROOT / "rl_baseline_v1.urdf").open("rb") as stream:
            for block in iter(lambda: stream.read(65536), b""):
                digest.update(block)
        with HASH_FILE.open(encoding="ascii") as stream:
            self.assertEqual(stream.readline(), digest.hexdigest())


if __name__ == "__main__":
    unittest.main()
