# Embedded firmware

Firmware for Pioneer's in-house motor controllers. Both projects build with
[PlatformIO](https://platformio.org/) (the VS Code extension or the `pio` CLI). CI skips this
folder, so build locally before you open a PR.

The CubeMars actuators on the arm and legs run their own firmware and are driven over CAN from
[`src/interfacing`](../interfacing). Nothing here runs on them.

| Folder | Target | What it is |
|---|---|---|
| [`STM32/`](STM32) | NUCLEO-G474RE (STM32G474) | FOC motor-controller firmware: MT6835 encoder, FDCAN, FreeRTOS, a custom bootloader |
| [`ESP32S3/gimbal_motor_testing/`](ESP32S3/gimbal_motor_testing) | Seeed XIAO ESP32S3 | Bench test: SimpleFOC open-loop velocity control of one gimbal BLDC motor |

## STM32

`STM32/platformio.ini` has three environments:

| Env | Framework | Sources | Purpose |
|---|---|---|---|
| `app` (default) | STM32Cube + FreeRTOS | `app/` | Application image, linked to start after the bootloader |
| `bootloader` | STM32Cube | `bootloader/` | 32 KiB bootloader at `0x08000000` that jumps to the app at `0x08008000` |
| `nucleo_g474re` | Arduino + SimpleFOC | `src/` | SimpleFOC prototype with the FDCAN and MT6835 drivers in `src/Drivers` |

The app image embeds the bootloader binary (`app/src/bootloader.S`), so build the bootloader
first:

```bash
cd src/embedded/STM32
pio run -e bootloader          # .pio/build/bootloader/bootloader.bin
python3 pad_bootloader.py      # pad it to exactly 32 KiB (0xFF fill)
pio run -e app -t upload       # build the app with the bootloader inside and flash over ST-Link
```

For the SimpleFOC prototype, run `pio run -e nucleo_g474re -t upload`, then
`pio device monitor` (115200 baud).

## ESP32S3 gimbal motor test

```bash
cd src/embedded/ESP32S3/gimbal_motor_testing
pio run -t upload && pio device monitor   # 115200 baud
```

Over serial, `T<rad/s>` sets the target velocity and `L<volts>` sets the voltage limit.
Current-limit the bench supply: the driver is set up for 12 V with a 3 V motor limit.
