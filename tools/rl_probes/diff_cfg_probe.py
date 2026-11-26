import pickle, sys
from isaaclab.app import AppLauncher
_launcher = AppLauncher({"headless": True, "device": "cuda:0"})
app = _launcher.app
import traceback
try:
    M=lambda m: open('/tmp/diff_stage.txt','a').write(m+'\n')
    M('start')
    import humanoid_rl_tasks  # noqa
    M('tasks imported')
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.utils.dict import class_to_dict
    TASK = "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-v0"
    default = class_to_dict(parse_env_cfg(TASK, device="cuda:0", num_envs=4))
    M('default built')
    saved = class_to_dict(pickle.load(open(sys.argv[1], "rb")))
    M('saved loaded')
    def walk(a, b, path=""):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                walk(a.get(k, "<missing>"), b.get(k, "<missing>"), f"{path}.{k}")
        elif a != b and not (isinstance(a, float) and isinstance(b, float) and abs(a - b) < 1e-12):
            sa, sb = str(a)[:110], str(b)[:110]
            if "object at 0x" in sa + sb or "class" in sa[:8]: return
            print(f"[DIFF] {path}: default={sa} saved={sb}", flush=True)
    walk(default, saved)
    print("[DIFF] DONE", flush=True); app.close() if hasattr(app, "close") else None

except BaseException:
    open('/tmp/diff_err.txt','w').write(traceback.format_exc()); raise
