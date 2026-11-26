"""Send one command to hanoi_server.py and print its JSON reply (image paths made absolute on this machine).
  python3 agent_cli.py state
  python3 agent_cli.py move left 0.5 0.18 0.93
  python3 agent_cli.py grip left close
  python3 agent_cli.py look film,top,wrist_left look_1
  python3 agent_cli.py caption Move 1: green from A to the tray
  python3 agent_cli.py done
"""
import json
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
a = sys.argv[1:]
c = {"op": a[0]}
if a[0] == "move":
    c.update(arm=a[1], x=float(a[2]), y=float(a[3]), z=float(a[4]))
elif a[0] == "grip":
    c.update(arm=a[1], how=a[2])
elif a[0] == "look":
    c.update(cams=a[1].split(","), name=a[2] if len(a) > 2 else "look")
elif a[0] == "caption":
    c["text"] = " ".join(a[1:])
s = socket.create_connection(("127.0.0.1", 7777), timeout=600)
s.sendall((json.dumps(c) + "\n").encode())
buf = b""
while not buf.endswith(b"\n"):
    buf += s.recv(65536)
r = json.loads(buf)
if "images" in r:
    r["images"] = [str(ROOT / p) for p in r["images"]]
print(json.dumps(r))
