# agent_control: a frontier LLM agent drives the 8dof arm directly

No trained policy. The agent (Claude, Codex, Gemini or any other agent that can run shell commands and open images)
reads a task brief and drives the arm one command at a time through `agent_cli.py`. The Isaac sim is paused between
commands, so the agent's thinking time does not show in the video. The first task is the Tower of Hanoi (`hanoi`
scene): three rings with a hole, three pegs, and a hand-over tray, because each arm points straight down only on its
own side.

## Run

```bash
./watod -t simulation_isaac
# in the container:
/workspace/isaaclab/isaaclab.sh -p src/simulation/agent_control/hanoi_server.py --run opus_1 --label "Claude Opus 5.5" --speedup 4 --headless
```

Then give the agent, running on the host from the repo root, [TASK_hanoi.md](TASK_hanoi.md) as its prompt.

- `--privileged`: `state` also returns every ring's pose. Without it (the default) the agent has only the fixed layout,
  its own fingertip points and grippers, and the cameras (film, top, both wrists, each with its calibration).
- `--speedup N`: the video plays N times faster than sim time.
- Output in `outputs/agent_control/RUN/`: `video.mp4`, the look images, and `log.jsonl` (every command, its reply, the
  true ring poses and a `solved` flag; the agent is told not to read it).

The server only exposes safe primitives: `move` is a straight fingertip line with the gripper pointing down, IK checked
at every waypoint, and refuses with `IK_UNREACHABLE`, `PATH_UNREACHABLE` or `BELOW_TABLE` instead of moving.

## Results (camera-only, 3 rings, one run each)

| | solved | commands | camera looks | failed grasps | agent time | sim time |
|---|---|---|---|---|---|---|
| Claude Fable 5.2 | yes | 115 | 13 | 0 | 5.4 min | 111 s |
| Claude Opus 5.5 | yes | 156 | 26 | 3 (all recovered) | 8.6 min | 127 s |

Both read which ring is on which peg from the images and planned the 7 moves with 3 tray hand-overs without a mistake.
The difference was in the physical details: Opus first gripped about 1 cm too high and closed on the peg twice, noticed
it from the gripper travel and retried lower; Fable got the heights right from the layout numbers the first time.
