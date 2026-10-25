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

try:
    import yaml
except ImportError:
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[4]
ASSET_ROOT = REPO_ROOT / "assets/whole_body_humanoid"
CACHE_ROOT = ASSET_ROOT / "usd"
HASH_FILE = CACHE_ROOT / ".asset_hash"


class AssetCacheContracts(unittest.TestCase):
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
        with (ASSET_ROOT / "whole_body_humanoid.urdf").open("rb") as stream:
            for block in iter(lambda: stream.read(65536), b""):
                digest.update(block)
        with HASH_FILE.open(encoding="ascii") as stream:
            self.assertEqual(stream.readline(), digest.hexdigest())


if __name__ == "__main__":
    unittest.main()
