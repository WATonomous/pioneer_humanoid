import time, numpy as np
import common as C
from synthetic_teleop import run_episode, TeleopSim
qs=[]
t=time.time()
ok, simt, notes, style = run_episode("peg_insert", 0, frame_sink=lambda a,s,task,e,q: qs.append((a,s,q)), record_every=4)
print("episode ok", ok, "sim s", simt, "frames", len(qs), "wall", time.time()-t, notes)
sim = TeleopSim("peg_insert", 0, cameras={c:(C.H,C.W) for c in C.CAMS})
import mujoco
r = mujoco.Renderer(sim.model, C.H, C.W)
for k in [0, len(qs)//3, 2*len(qs)//3, len(qs)-1]:
    sim.render_at(qs[k][2])
    out={}
    for cam in C.CAMS:
        r.update_scene(sim.data, cam); rgb=r.render()
        r.enable_depth_rendering(); r.update_scene(sim.data, cam); d=r.render(); r.disable_depth_rendering()
        out[cam]=(rgb,d)
        print(k, cam, rgb.shape, d.min().round(3), np.median(d).round(3), d.max().round(3))
    np.savez(C.DATA/f"smoke_{k}.npz", **{f"{c}_rgb":out[c][0] for c in C.CAMS}, **{f"{c}_depth":out[c][1] for c in C.CAMS})
