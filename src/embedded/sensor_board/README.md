# Sensor board firmware

## Setup and build

From the repository root, initialize the pinned dependencies:

```sh
git submodule update --init --recursive
```

Then build from `src/embedded/sensor_board`:

```sh
pio run -e nucleo_g474re
```

## FreeRTOS

`lib/FreeRTOS-Kernel` is an upstream Git submodule pinned to commit
`e3a0e3e86aa196b4b2e76b851fe152f426f091f7`. This revision matches the
previously copied kernel files, including the Cortex-M4F port and heap allocator.

`freertos_build.py` selects only the kernel sources, `portable/GCC/ARM_CM4F`
port, and `portable/MemMang/heap_4.c`. PlatformIO's automatic library scanning
ignores this dependency to avoid building other architecture ports and heaps.
Keep project configuration in `include/FreeRTOSConfig.h`; do not edit the
submodule for board-specific configuration.

After switching branches or pulling a dependency update, rerun the submodule
initialization command above. A fresh clone can use `git clone --recurse-submodules`.
