#!/usr/bin/env python3
"""RTC policy server for a LeRobot flow-matching checkpoint (SmolVLA/pi0/pi0.5).

Runs in the Python 3.12 venv (/opt/vla_env), because lerobot needs Python >= 3.12
while Isaac Sim is Python 3.10. Isaac Lab connects via lerobot_eval_rtc_remote.py.

Protocol: length-prefixed pickles over TCP (stdlib only, neither interpreter has zmq).
  {"cmd": "ping" | "reset" | "act", "state": (6,) float32 raw-units, "images": {cam: HxWx3 uint8}}
  -> {"ok": True, "action": (6,) float32 raw-units}
"""

import argparse
import pickle
import socket
import struct

import numpy as np
import torch

import lerobot.policies  # noqa: F401  (populates the policy registry)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig

from humanoid_il.rtc_driver import RTCDrivenPolicy
from humanoid_il.so101_sim import SO101_LEADER_KEYS

CAMERA_KEYS = ("ego", "external_D455")


class DummyDatasetMeta:
    def __init__(self, features, robot_type):
        self.features = features
        self.stats = {}
        self.robot_type = robot_type


def recv_msg(conn):
    header = b""
    while len(header) < 8:
        chunk = conn.recv(8 - len(header))
        if not chunk:
            return None
        header += chunk
    (n,) = struct.unpack("!Q", header)
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            return None
        buf.extend(chunk)
    return pickle.loads(bytes(buf))


def send_msg(conn, obj):
    data = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack("!Q", len(data)) + data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy_path", required=True)
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--execution_horizon", type=int, default=10)
    p.add_argument("--max_guidance_weight", type=float, default=10.0)
    p.add_argument("--lang_instruction", default="Pick up the vial and place it in the rack")
    args = p.parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.pretrained_path = args.policy_path
    cfg.device = args.device
    cfg.rtc_config = RTCConfig(
        enabled=True, execution_horizon=args.execution_horizon, max_guidance_weight=args.max_guidance_weight
    )

    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": list(SO101_LEADER_KEYS)},
        "action": {"dtype": "float32", "shape": (6,), "names": list(SO101_LEADER_KEYS)},
    }
    for cam in CAMERA_KEYS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"],
        }
    policy = make_policy(cfg, ds_meta=DummyDatasetMeta(features, "so101_follower"))
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=args.policy_path,
        dataset_stats={},
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    driver = RTCDrivenPolicy(
        policy=policy, preprocessor=pre, postprocessor=post, task_description=args.lang_instruction,
        robot_type="so101_follower", execution_horizon=args.execution_horizon,
    )
    print("[SERVER] policy ready", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[SERVER] listening on {args.host}:{args.port}", flush=True)

    while True:
        conn, _ = srv.accept()
        print("[SERVER] client connected", flush=True)
        with conn:
            while True:
                req = recv_msg(conn)
                if req is None:
                    break
                try:
                    cmd = req["cmd"]
                    if cmd == "ping":
                        send_msg(conn, {"ok": True})
                    elif cmd == "reset":
                        driver.reset()
                        send_msg(conn, {"ok": True})
                    elif cmd == "act":
                        frame = {"observation.state": np.asarray(req["state"], dtype=np.float32)}
                        for cam, img in req["images"].items():
                            frame[f"observation.images.{cam}"] = np.asarray(img, dtype=np.uint8)
                        with torch.no_grad():
                            action = driver.get_action(frame)
                        send_msg(conn, {"ok": True, "action": action.detach().cpu().numpy().reshape(-1)[:6]})
                    else:
                        send_msg(conn, {"ok": False, "error": f"unknown cmd {cmd}"})
                except Exception as e:  # keep serving; report to the client
                    import traceback

                    traceback.print_exc()
                    send_msg(conn, {"ok": False, "error": repr(e)})
        print("[SERVER] client disconnected", flush=True)


if __name__ == "__main__":
    main()
