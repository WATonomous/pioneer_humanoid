"""Build the pinned FreeRTOS kernel with the STM32G474 port and one heap."""
from pathlib import Path

Import("env")

kernel_dir = Path(env.subst("$PROJECT_DIR")) / "lib" / "FreeRTOS-Kernel"
if not (kernel_dir / "tasks.c").is_file():
    raise RuntimeError(
        "FreeRTOS submodule is missing. Run: git submodule update --init --recursive"
    )

# An upstream checkout includes many mutually exclusive architecture ports.
# Keep the selection here so the submodule itself stays unmodified.
sources = [
    "croutine.c",
    "event_groups.c",
    "list.c",
    "queue.c",
    "stream_buffer.c",
    "tasks.c",
    "timers.c",
    "portable/GCC/ARM_CM4F/port.c",
    "portable/MemMang/heap_4.c",
]

env.BuildSources(
    "$BUILD_DIR/freertos",
    str(kernel_dir),
    src_filter=["-<*>"] + [f"+<{source}>" for source in sources],
)
