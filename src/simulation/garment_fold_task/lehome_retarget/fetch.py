"""Download what the retarget needs from LeHome's Hugging Face datasets (Apache-2.0) into $LEHOME_ROOT (default
~/Desktop/lehome_ref): their merged demos (state/action parquet + garment poses, no videos), the SO-101 URDF, all
Release garments and the apartment scene (also copied into vendor_assets/scenes/marble, gitignored)."""
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(os.environ.get("LEHOME_ROOT", Path.home() / "Desktop/lehome_ref"))
for k in ("top_short", "top_long", "pant_short", "pant_long"):
    for f in ("data/chunk-000/file-000.parquet", "meta/garment_info.json", "meta/episodes/chunk-000/file-000.parquet"):
        hf_hub_download("lehome/dataset_challenge_merged", f"{k}_merged/{f}", repo_type="dataset", local_dir=ROOT / "data")
snapshot_download("lehome/asset_challenge", repo_type="dataset", local_dir=ROOT / "assets",
                  allow_patterns=["robots/so101_new_calib.urdf", "objects/Challenge_Garment/Release/**",
                                  "scenes/marble/Scene_00_Apartment.usd", "scenes/marble/Scene_00_Apartment.usdz"])
dst = Path(__file__).resolve().parents[1] / "vendor_assets/scenes/marble"
dst.mkdir(parents=True, exist_ok=True)
for f in ("Scene_00_Apartment.usd", "Scene_00_Apartment.usdz"):
    shutil.copy(ROOT / "assets/scenes/marble" / f, dst / f)
