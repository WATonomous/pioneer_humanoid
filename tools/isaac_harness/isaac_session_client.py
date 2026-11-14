#!/usr/bin/env python3
"""Host-side client for isaac_session_daemon.py.

The queue directory lives under the repo (tools/isaac_harness/.session/),
which is bind-mounted read-write into the container at
/workspace/humanoid/tools/isaac_harness/.session — so commands/responses
pass between host and daemon as plain files, no docker cp needed.

Usage (each subcommand sends one command and prints the JSON response):

    isaac_session_client.py ping
    isaac_session_client.py spawn_primitive --name Obj1 --shape cuboid \
        --size 0.1,0.1,0.1 --pos 0,0,0.5 --color 1,0,0
    isaac_session_client.py spawn_usd --name Robot --usd-path /workspace/... \
        --pos 0,0,0 --articulation
    isaac_session_client.py set_pose --name Obj1 --pos 0,0,1
    isaac_session_client.py query --name Obj1
    isaac_session_client.py list
    isaac_session_client.py step --n 30
    isaac_session_client.py screenshot --eye 2,2,2 --target 0,0,0 --out shot.png
    isaac_session_client.py remove --name Obj1
    isaac_session_client.py shutdown

Or import send_command() directly from Python for tighter control.
"""
import argparse
import json
import os
import sys
import time
import uuid

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
QUEUE_DIR = os.path.join(REPO_ROOT, "tools", "isaac_harness", ".session")
CMDS_DIR = os.path.join(QUEUE_DIR, "cmds")
RESP_DIR = os.path.join(QUEUE_DIR, "responses")


def send_command(cmd: dict, timeout: float = 30.0) -> dict:
    os.makedirs(CMDS_DIR, exist_ok=True)
    os.makedirs(RESP_DIR, exist_ok=True)

    cmd_id = str(uuid.uuid4())
    cmd["id"] = cmd_id
    tmp_path = os.path.join(CMDS_DIR, f"{cmd_id}.cmd.json.tmp")
    final_path = os.path.join(CMDS_DIR, f"{cmd_id}.cmd.json")
    with open(tmp_path, "w") as f:
        json.dump(cmd, f)
    os.rename(tmp_path, final_path)

    resp_path = os.path.join(RESP_DIR, f"{cmd_id}.response.json")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(resp_path):
            try:
                with open(resp_path) as f:
                    response = json.load(f)
                os.remove(resp_path)
                return response
            except (json.JSONDecodeError, OSError):
                time.sleep(0.05)
                continue
        time.sleep(0.05)
    return {"ok": False, "error": f"timed out after {timeout}s waiting for response to {cmd.get('cmd')}"}


def _floats(s: str) -> list:
    return [float(x) for x in s.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ping")
    sub.add_parser("list")
    sub.add_parser("shutdown")

    p = sub.add_parser("spawn_primitive")
    p.add_argument("--name", required=True)
    p.add_argument("--shape", default="cuboid", choices=["cuboid", "sphere", "cylinder", "cone"])
    p.add_argument("--size", type=_floats, default=[0.1, 0.1, 0.1])
    p.add_argument("--pos", type=_floats, default=[0, 0, 0])
    p.add_argument("--color", type=_floats, default=[0.8, 0.8, 0.8])
    p.add_argument("--mass", type=float, default=0.1)
    p.add_argument("--static", action="store_true", help="not affected by physics")

    p = sub.add_parser("spawn_usd")
    p.add_argument("--name", required=True)
    p.add_argument("--usd-path", required=True)
    p.add_argument("--pos", type=_floats, default=[0, 0, 0])
    p.add_argument("--rot", type=_floats, default=[1, 0, 0, 0])
    p.add_argument("--scale", type=_floats, default=[1, 1, 1])
    p.add_argument("--articulation", action="store_true")
    p.add_argument("--apply-physics", action="store_true",
                   help="apply RigidBodyAPI/CollisionAPI/MassAPI to a bare-mesh asset with no "
                   "physics baked in (e.g. YCB props) — leave off for assets that already "
                   "have their own, like the vial/rack")
    p.add_argument("--mass", type=float, default=0.2)

    p = sub.add_parser("spawn_bimanual_arm",
                       help="spawn this repo's pioneer_bimanual_arm_8dof robot with its real actuator config")
    p.add_argument("--name", required=True)
    p.add_argument("--pos", type=_floats, default=[0, 0, 0])
    p.add_argument("--rot", type=_floats, default=[1, 0, 0, 0])

    p = sub.add_parser("spawn_visual",
                       help="spawn a USD asset as a pure visual reference, no physics wrapper "
                            "(for assets with no baked-in RigidBodyAPI, e.g. YCB props)")
    p.add_argument("--name", required=True)
    p.add_argument("--usd-path", required=True)
    p.add_argument("--pos", type=_floats, default=[0, 0, 0])
    p.add_argument("--rot", type=_floats, default=[1, 0, 0, 0])
    p.add_argument("--scale", type=_floats, default=[1, 1, 1])

    p = sub.add_parser("list_dir", help="list a Nucleus (or local) directory via omni.client")
    p.add_argument("--path", required=True)

    p = sub.add_parser("remove")
    p.add_argument("--name", required=True)

    p = sub.add_parser("set_pose")
    p.add_argument("--name", required=True)
    p.add_argument("--pos", type=_floats, default=None)
    p.add_argument("--rot", type=_floats, default=None)

    p = sub.add_parser("query")
    p.add_argument("--name", required=True)

    p = sub.add_parser("joint_state", help="per-joint positions/velocities for an articulated object")
    p.add_argument("--name", required=True)

    p = sub.add_parser("step")
    p.add_argument("--n", type=int, default=1)

    p = sub.add_parser("screenshot")
    p.add_argument("--eye", type=_floats, default=[2, 2, 2])
    p.add_argument("--target", type=_floats, default=[0, 0, 0])
    p.add_argument("--out", required=True, help="host-side output path")

    p = sub.add_parser("bbox", help="world-space axis-aligned bounding box of a prim")
    p.add_argument("--name", required=True)

    p = sub.add_parser("overlap", help="AABB overlap check between two prims")
    p.add_argument("--name-a", required=True)
    p.add_argument("--name-b", required=True)

    p = sub.add_parser("distance", help="center-to-center distance between two prims")
    p.add_argument("--name-a", required=True)
    p.add_argument("--name-b", required=True)

    sub.add_parser("scene_tree", help="full /World prim hierarchy with types")

    args = parser.parse_args()
    cmd = {"cmd": args.cmd}

    if args.cmd == "spawn_primitive":
        cmd.update(name=args.name, shape=args.shape, size=args.size, pos=args.pos,
                   color=args.color, mass=args.mass, dynamic=not args.static)
    elif args.cmd == "spawn_usd":
        cmd.update(name=args.name, usd_path=args.usd_path, pos=args.pos, rot=args.rot, scale=args.scale,
                   articulation=args.articulation, apply_physics=args.apply_physics, mass=args.mass)
    elif args.cmd == "spawn_bimanual_arm":
        cmd.update(name=args.name, pos=args.pos, rot=args.rot)
    elif args.cmd == "spawn_visual":
        cmd.update(name=args.name, usd_path=args.usd_path, pos=args.pos, rot=args.rot, scale=args.scale)
    elif args.cmd == "list_dir":
        cmd.update(path=args.path)
    elif args.cmd == "remove":
        cmd.update(name=args.name)
    elif args.cmd == "set_pose":
        cmd.update(name=args.name, pos=args.pos, rot=args.rot)
    elif args.cmd == "query":
        cmd.update(name=args.name)
    elif args.cmd == "joint_state":
        cmd.update(name=args.name)
    elif args.cmd == "step":
        cmd.update(n=args.n)
    elif args.cmd == "screenshot":
        # translate host --out path to the container-visible path under the
        # same bind-mounted repo root
        container_out = os.path.join(
            "/workspace/humanoid/tools/isaac_harness/.session/shots",
            os.path.basename(args.out),
        )
        cmd.update(eye=args.eye, target=args.target, out_path=container_out)
    elif args.cmd == "bbox":
        cmd.update(name=args.name)
    elif args.cmd in ("overlap", "distance"):
        cmd.update(name_a=args.name_a, name_b=args.name_b)

    response = send_command(cmd)

    if args.cmd == "screenshot" and response.get("ok"):
        host_container_shot = os.path.join(QUEUE_DIR, "shots", os.path.basename(args.out))
        if os.path.exists(host_container_shot):
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            os.replace(host_container_shot, args.out)
            response["out_path"] = args.out

    print(json.dumps(response))
    sys.exit(0 if response.get("ok") else 1)


if __name__ == "__main__":
    main()
