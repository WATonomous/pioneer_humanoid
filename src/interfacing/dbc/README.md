# DBC Interface

`humanoid.dbc` defines the CAN messages and signals `can_node` encodes and decodes: servo-mode
commands and feedback (also the AK's MIT feedback), plus the GL II's MIT command frame below.

## MITControlCmd

A **standard 11-bit** frame (`BO_ 0`); every servo message is extended. A GL II in MIT mode only
listens to standard frames. Its signals are raw MIT codes (`pos(16) vel(12) kp(12) kd(12)
t_ff(12)`) that `can_node` packs from physical units via `can/config/mit_profiles.yaml`. MIT
feedback is decoded in `can_node`, not here.

## Two ways to decode DBC

Turning raw CAN bytes into named signals can be done statically or dynamically. This repo
uses the dynamic path on the ROS side:

| | Static (`decode.c`) | Dynamic (`can_node` + `libdbcppp`) ← used here |
|---|---|---|
| DBC read | compiled into C ahead of time | `humanoid.dbc` loaded from file at startup |
| Change the DBC | regenerate + recompile | edit `.dbc`, restart the node |
| Cost | tiny, no runtime parsing | needs the lib + parses on start |
| Fits | bare-metal firmware (STM32/ESP32) | Linux / ROS host |

`can_node` links `libdbcppp.so` and installs `humanoid.dbc` into its package share, then
loads and decodes against it at runtime (you'll see `Loaded DBC message: ...` on startup).
So it **never uses `decode.c`** — that static decoder is only for embedded boards that
can't run libdbcppp.

## Usage

`decode.c` is generated on demand from `humanoid.dbc`, not committed. To regenerate, enter
the docker container and run:
`dbcparser dbc2 --dbc=dbc/humanoid.dbc --format=C >> decode.c`

