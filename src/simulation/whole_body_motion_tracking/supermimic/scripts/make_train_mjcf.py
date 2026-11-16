"""python make_train_mjcf.py <retarget.xml> <out.xml>

Training MJCF for SuperMimic on mjlab, from the retarget model (floating pelvis, fists, real elbow range):
no gravity compensation (the asset gives every body gravcomp="1", which would make the arms weightless), no floor or
lights (mjlab adds its terrain), and each collision box named "<body>_collision" (feet: l_foot_collision, ...), which
is how mjlab's CollisionCfg picks geoms.
"""
import sys
import xml.etree.ElementTree as ET

src, out = sys.argv[1], sys.argv[2]
root = ET.parse(src).getroot()
world = root.find("worldbody")
for el in list(world):
    if el.tag == "light" or (el.tag == "geom" and el.get("type") == "plane"):
        world.remove(el)
for body in root.iter("body"):
    body.attrib.pop("gravcomp", None)
    hitboxes = [g for g in body.findall("geom") if g.get("group") == "3"]
    for i, g in enumerate(hitboxes):
        g.set("name", f"{body.get('name')}_collision" + (f"{i}" if i else ""))
ET.ElementTree(root).write(out)
print("wrote", out)
