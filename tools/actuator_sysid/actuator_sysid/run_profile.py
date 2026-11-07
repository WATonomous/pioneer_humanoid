"""Run one bench profile on one motor and save the run folder (log.npz + meta.json).

Dry run against the fake motor (no hardware, works today):
    conda activate sysid
    python -m actuator_sysid.run_profile --motor AK60-39 --id 40 --profile step --fake \\
        --fixture "dry run" --set size=0.05

Real hardware is locked until the Task 10 reviewer checkpoint (D11, bus.LIVE_HARDWARE_ENABLED).
After that: --interface socketcan --channel can0 --live, then type the motor id when asked.

Options:
  --set NAME=VALUE        profile parameter (repeatable), e.g. --set size=0.05 --set speeds=0.1,0.2
  --limit NAME=VALUE      bench limit (repeatable); raising above a default needs --limit-reason
  --fixture/--firmware/--voltage/--note   go into meta.json
Exit codes: 0 done (or stopped at abort_distance), 2 safety fault, 3 refused before start, 130 Ctrl-C.
"""
import argparse
import dataclasses
import sys
import time
from pathlib import Path

from .bus import CanBus, HardwareLocked, confirm_live
from .fake_motor import FakeMotor
from .logger import DEFAULT_OUT, Recorder, next_run_dir, save_run
from .motors import MOTORS, make_motor, short_name
from .profiles import PROFILES, check_profile, make_profile, run
from .safety import BenchLimits, Safety, SafetyFault


# --- Shared with bringup.py ---------------------------------------------------------------------
def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--motor", required=True, choices=sorted(MOTORS))
    ap.add_argument("--id", type=int, required=True, help="CAN id of the motor (0..255)")
    ap.add_argument("--interface", default="virtual", help="virtual (default) or socketcan")
    ap.add_argument("--channel", default="sysid", help="virtual channel name, vcan0, or (after D11) can0")
    ap.add_argument("--live", action="store_true", help="real hardware (locked until the Task 10 review)")
    ap.add_argument("--fake", action="store_true", help="start a fake motor on the same virtual bus")
    ap.add_argument("--fake-position", type=float, default=0.3, help="fake motor start position (rad)")
    ap.add_argument("--limit", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--limit-reason", default=None)
    ap.add_argument("--fixture", required=True, help='e.g. "clamped, 0.3 m arm, 1 kg"')
    ap.add_argument("--firmware", default="unknown")
    ap.add_argument("--voltage", type=float, default=None, help="supply voltage read on the PSU (V)")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--yes", action="store_true", help="don't ask before moving (not allowed with --live)")


def parse_pairs(pairs: list[str], what: str) -> dict[str, str]:
    out = {}
    for item in pairs:
        name, sep, value = item.partition("=")
        if not sep or not name:
            raise ValueError(f"{what} must look like NAME=VALUE, got {item!r}")
        out[name.strip()] = value.strip()
    return out


def setup(args):
    """Validate everything that can be checked before touching the bus. Returns (spec, motor, limits)."""
    spec = MOTORS[args.motor]
    motor = make_motor(spec, args.id)
    limits = BenchLimits().with_overrides(
        {k: float(v) for k, v in parse_pairs(args.limit, "--limit").items()}, args.limit_reason)
    limits.check_against(spec)
    if args.live and args.yes:
        raise ValueError("--yes is not allowed with --live: every live step needs the operator")
    return spec, motor, limits


def open_bus(args, spec):
    """Confirm (live), open the bus, start the fake motor if asked. Returns (bus, fake or None)."""
    if args.live:
        confirm_live(args.id)
    bus = CanBus(args.interface, args.channel, live=args.live)
    fake = None
    if args.fake:
        if bus.kind == "hardware":
            bus.close()
            raise HardwareLocked("--fake is only for virtual buses")
        fake = FakeMotor(spec, args.id, interface=args.interface, channel=args.channel,
                         position=args.fake_position).start()
        time.sleep(0.05)
    return bus, fake


def base_meta(args, spec, limits, bus, fake) -> dict:
    return {
        "motor": {"name": spec.name, "short": short_name(spec), "id": args.id, "protocol": spec.protocol,
                  "spec": dataclasses.asdict(spec)},
        "firmware": args.firmware, "fixture": args.fixture, "voltage_v": args.voltage, "note": args.note,
        "limits": limits.to_dict(), "limits_raised": limits.raised_above_defaults(),
        "limit_reason": args.limit_reason,
        "bus": {"interface": args.interface, "channel": args.channel, "kind": bus.kind,
                "timestamp_source": bus.timestamp_source},
        "dry_run": bus.kind != "hardware",
        "fake_motor": None if fake is None else {"position": args.fake_position, "inertia": fake.inertia,
                                                 "coulomb": fake.coulomb, "viscous": fake.viscous},
        "command_line": sys.argv,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def ask(prompt: str, args) -> None:
    if args.yes:
        print(prompt + " [--yes]")
        return
    input(prompt + " Press Enter to continue, Ctrl-C to stop. ")


# --- This script ------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--profile", required=True, choices=sorted(PROFILES))
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="profile parameter")
    args = ap.parse_args(argv)

    try:
        spec, motor, limits = setup(args)
        profile = make_profile(args.profile, **parse_pairs(args.set, "--set"))
        check_profile(profile, limits, spec, start=0.0)  # again in run() with the live start position
        bus, fake = open_bus(args, spec)
    except (ValueError, KeyError, HardwareLocked) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 3

    run_dir = next_run_dir(args.out, short_name(spec), args.id, profile.name)
    rec = Recorder()
    meta = base_meta(args, spec, limits, bus, fake)
    meta.update({"profile": profile.params(), "gains": {"kp": profile.kp, "kd": profile.kd}})
    result, code, safety = None, 0, None
    try:
        print(f"{spec.name} id {args.id} on {args.interface}:{args.channel} ({bus.kind}), profile "
              f"{profile.name} {profile.duration:.1f} s, limits {limits.to_dict()}")
        ask("Ready to arm the motor and run.", args)
        with Safety(bus, motor, limits) as safety:
            safety.listeners.append(rec)
            start = safety.arm()
            print(f"start position {start:.4f} rad")
            result = run(safety, profile, recorder=rec)
        print(f"done: {result.ended}, loop {result.ticker}")
    except SafetyFault as e:
        print(f"SAFETY FAULT: {e} (motor damped and disabled)", file=sys.stderr)
        code = 2
    except KeyboardInterrupt:
        print("stopped by Ctrl-C (motor damped and disabled)", file=sys.stderr)
        code = 130
    except SystemExit as e:  # SIGTERM / SIGHUP (safety.py turns them into SystemExit)
        print(f"stopped by signal (motor damped and disabled), exit {e.code}", file=sys.stderr)
        code = e.code if isinstance(e.code, int) else 1
    except ValueError as e:  # profile doesn't fit at the live start position
        print(f"refused: {e}", file=sys.stderr)
        code = 3
    finally:
        meta.update({
            "start_position": None if safety is None else safety.start_position,
            "result": {"ended": None if result is None else result.ended,
                       "fault": None if safety is None else safety.fault_reason,
                       "disable_acknowledged": None if safety is None else safety.disable_acknowledged,
                       "exit_code": code},
            "loop": None if result is None else result.ticker,
        })
        save_run(run_dir, rec, meta)
        bus.close()
        if fake is not None:
            fake.stop()
        print(f"saved {run_dir}")
    return code


if __name__ == "__main__":
    sys.exit(main())
