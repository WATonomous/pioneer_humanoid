"""Download the MDL materials and textures a Vegetation tree USD references (into ./trees/materials)."""
import re, subprocess, sys
from pathlib import Path
from pxr import UsdUtils
B = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Vegetation/Plant_Tropical/"
root = Path("tropical")
for usd in sys.argv[1:]:
    _, _, unresolved = UsdUtils.ComputeAllDependencies(str(root / usd))
    for p in [q for q in unresolved if q.startswith(str(root.resolve()))]:
        rel = Path(p).relative_to(root.resolve())
        dst = root / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["curl", "-sfL", "-o", str(dst), B + str(rel)])
        if dst.suffix == ".mdl" and dst.exists():
            for tex in re.findall(r'texture_2d\("\./([^"]+)"', dst.read_text()):
                tdst = dst.parent / tex
                if not tdst.exists():
                    tdst.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(["curl", "-sfL", "-o", str(tdst), B + "materials/" + tex])
    print(usd, "ok")
