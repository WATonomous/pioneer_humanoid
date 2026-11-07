"""Bench bring-up for one motor, in Task 10's order: read-only -> kp = 0 -> low-kp hold.

DRY RUN ONLY until the Task 10 reviewer checkpoint (D11): real adapters are refused by bus.py
(LIVE_HARDWARE_ENABLED = False). Dry run against the fake motor:
    conda activate sysid
    python -m actuator_sysid.bringup --motor AK80-64 --id 50 --fake --fixture "dry run"

Stages (each asks the operator first unless --yes; --yes is refused with --live):
  1. read-only    listen --read-time s, send nothing. AK 3.0 must stream 0x29 (else stop: check
                  power, wiring, id, R-Link feedback rate). AK 2.0 replies only to commands, so
                  silence is expected here.
  2. kp = 0       arm (AK 2.0: enter MIT + zero-torque probes) and send kp = kd = t = 0 for
                  --kp0-time s: the motor can't produce torque. AK80-64 (no absolute position):
                  zero it first at the mark (--zero for an AK 3.0 temporary origin). Turn the output
                  90 deg by hand: the printout must read ~90 deg with the expected sign.
                  Limits here watch the readings only (window +-3.2 rad, 3 rad/s), logged with a reason.
  3. low-kp hold  default bench limits; hold the live position with kp --hold-kp, kd --hold-kd for
                  --hold-time s. Push gently: it should resist softly and come back.
Then damp 1 s and disable (safety.py). Everything goes into one run folder <motor>_<id>_bringup_<n>.
"""
import argparse
import math
import sys
import threading
import time

from .bus import HardwareLocked
from .logger import Recorder, Ticker, next_run_dir, save_run
from .motors import short_name
from .protocol import MitCommand
from .run_profile import add_common_args, base_meta, open_bus, setup
from .safety import Safety, SafetyFault, check_command

KP0_OVERRIDES = {"position_window": 3.2, "max_speed": 3.0}
KP0_REASON = ("bring-up kp = 0 stage: commands are kp = kd = t = 0 (no torque possible); wider window and "
              "speed only so a 90 deg hand turn doesn't trip the reading checks")
ZERO_TOLERANCE = 0.01  # rad: reading right after zeroing


def _print_every(period: float):
    last = [0.0]

    def due() -> bool:
        now = time.monotonic()
        if now - last[0] >= period:
            last[0] = now
            return True
        return False
    return due


def stage_read_only(safety: Safety, seconds: float) -> dict:
    n, first, fb = 0, None, None
    safety.poll()  # frames queued before the stage started don't count towards the rate
    end = time.monotonic() + seconds
    due = _print_every(0.5)
    while time.monotonic() < end:
        new = safety.poll()
        n += len(new)
        if new:
            fb = new[-1]
            first = first if first is not None else time.monotonic()
            if due():
                print(f"  read-only: {math.degrees(fb.position):9.3f} deg  {fb.velocity:7.3f} rad/s  "
                      f"{fb.torque:7.3f} Nm  {fb.temperature} degC  error {fb.error}")
        time.sleep(0.002)
    result = {"frames": n, "rate_hz": n / seconds, "sent": safety.armed}
    if fb is not None:
        result.update({"position": fb.position, "temperature": fb.temperature, "error": fb.error,
                       "motor_fault": safety.motor.fault(fb)})
    return result


def stage_kp0(safety: Safety, seconds: float, period: float, recorder) -> dict:
    start = safety.start_position
    cmd = MitCommand(start, 0.0, 0.0, 0.0, 0.0)
    ticker, due = Ticker(period), _print_every(0.5)
    max_delta = last_delta = 0.0
    while ticker.elapsed < seconds:
        late = ticker.wait()
        safety.poll()
        recorder.on_tick(ticker.elapsed, late)
        safety.command(cmd)
        if safety.feedback is not None:
            last_delta = safety.feedback.position - start
            max_delta = max(max_delta, abs(last_delta))
            if due():
                print(f"  kp = 0: {math.degrees(last_delta):8.2f} deg from start  "
                      f"({safety.feedback.velocity:6.3f} rad/s, {safety.feedback.torque:6.3f} Nm)")
    return {"seconds": seconds, "max_abs_delta_deg": math.degrees(max_delta),
            "final_delta_deg": math.degrees(last_delta), "loop": ticker.stats()}


def stage_hold(safety: Safety, seconds: float, kp: float, kd: float, period: float, recorder) -> dict:
    start = safety.start_position
    cmd = MitCommand(start, 0.0, kp, kd, 0.0)
    check_command(cmd, safety.limits, safety.motor.spec, start)  # fail before the first frame
    ticker, due = Ticker(period), _print_every(0.5)
    max_err = max_torque = 0.0
    while ticker.elapsed < seconds:
        late = ticker.wait()
        safety.poll()
        recorder.on_tick(ticker.elapsed, late)
        safety.command(cmd)
        fb = safety.feedback
        if fb is not None:
            max_err, max_torque = max(max_err, abs(start - fb.position)), max(max_torque, abs(fb.torque))
            if due():
                print(f"  hold: error {math.degrees(start - fb.position):7.3f} deg  torque {fb.torque:6.3f} Nm")
    return {"seconds": seconds, "kp": kp, "kd": kd, "max_track_error_rad": max_err, "max_abs_torque_nm": max_torque,
            "loop": ticker.stats()}


def ask_operator(prompt: str, args, input_fn) -> str:
    """Ask while nothing is armed (blocking is fine: nothing is being sent)."""
    if args.yes:
        print(prompt + " [--yes]")
        return "skipped (--yes)"
    return input_fn(prompt + " ").strip()


def ask_while_armed(safety: Safety, cmd: MitCommand, prompt: str, args, input_fn, recorder) -> str:
    """Ask the operator while the motor is armed: the question runs in a thread and this loop keeps
    sending `cmd` at the loop rate, so the watchdog and the motor's CAN timeout never see a gap."""
    if args.yes:
        print(prompt + " [--yes]")
        return "skipped (--yes)"
    box = {}
    thread = threading.Thread(target=lambda: box.setdefault("answer", input_fn(prompt + " ")), daemon=True)
    thread.start()
    ticker = Ticker(args.period)
    while thread.is_alive():
        late = ticker.wait()
        safety.poll()
        recorder.on_tick(ticker.elapsed, late)
        safety.command(cmd)
    return str(box.get("answer", "")).strip()


def main(argv=None, input_fn=input) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--read-time", type=float, default=2.0)
    ap.add_argument("--kp0-time", type=float, default=20.0)
    ap.add_argument("--hold-time", type=float, default=10.0)
    ap.add_argument("--hold-kp", type=float, default=2.0)
    ap.add_argument("--hold-kd", type=float, default=0.2)
    ap.add_argument("--zero", action="store_true", help="AK 3.0: set a temporary origin (AK 2.0 always zeros)")
    ap.add_argument("--period", type=float, default=0.002)
    args = ap.parse_args(argv)

    try:
        spec, motor, limits = setup(args)
        kp0_limits = limits.with_overrides(KP0_OVERRIDES, KP0_REASON)
        kp0_limits.check_against(spec)
        check_command(MitCommand(0.0, 0.0, args.hold_kp, args.hold_kd, 0.0), limits, spec, 0.0)
        bus, fake = open_bus(args, spec)
    except (ValueError, KeyError, HardwareLocked) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 3

    zero = spec.protocol == "ak2" or args.zero
    print(("DRY RUN" if bus.kind != "hardware" else "LIVE HARDWARE")
          + f": {spec.name} id {args.id} on {args.interface}:{args.channel}")
    run_dir = next_run_dir(args.out, short_name(spec), args.id, "bringup")
    rec = Recorder()
    meta = base_meta(args, spec, limits, bus, fake)
    meta["bringup"] = {"kp0_limits": kp0_limits.to_dict(), "kp0_reason": KP0_REASON, "zero": zero,
                       "hold": {"kp": args.hold_kp, "kd": args.hold_kd}}
    stages, code, safety = {}, 0, None
    try:
        with Safety(bus, motor, limits, period=args.period) as safety:
            safety.listeners.append(rec)

            ask_operator("Stage 1, read-only: nothing will be sent. Press Enter.", args, input_fn)
            safety.note("stage read_only")
            stages["read_only"] = r = stage_read_only(safety, args.read_time)
            print(f"  {r['frames']} feedback frames ({r['rate_hz']:.0f} Hz)")
            if motor.streams_feedback and r["frames"] == 0:
                print("  no feedback from an AK 3.0 motor: check power, CAN wiring/termination, id, "
                      "R-Link feedback rate. Stopping.", file=sys.stderr)
                stages["stopped"] = "no feedback in read-only"
                return 2
            if not motor.streams_feedback and r["frames"]:
                print("  note: this AK 2.0 motor sent feedback without a command (not expected from the manual)")
            if r.get("motor_fault"):
                print(f"  motor reports a fault: {r['motor_fault']}. Stopping.", file=sys.stderr)
                stages["stopped"] = r["motor_fault"]
                return 2

            # Everything that needs the operator before arming is asked now: once armed, the loop
            # must keep sending (watchdog 50 ms, motor CAN timeout).
            ask_operator("Stage 2, kp = 0: the motor will be enabled with zero torque. Press Enter.", args, input_fn)
            if zero:
                ask_operator("Move the output to its zero mark and let go. Press Enter to zero it.", args, input_fn)
            safety.arm()
            if zero:
                p0 = safety.set_zero()
                stages["zero"] = {"position_after": p0, "ok": abs(p0) <= ZERO_TOLERANCE}
                print(f"  zero set: reads {p0:.5f} rad" + ("" if abs(p0) <= ZERO_TOLERANCE else "  NOT ~0, check"))
            start = safety.begin_stage(kp0_limits, "kp0")
            print(f"  turn the output +90 deg by hand within {args.kp0_time:g} s, then back")
            stages["kp0"] = stage_kp0(safety, args.kp0_time, args.period, rec)
            limp = MitCommand(start, 0.0, 0.0, 0.0, 0.0)
            stages["kp0"]["operator_90deg_reads_90"] = ask_while_armed(
                safety, limp, "Did +90 deg read ~+90 deg (y/n)?", args, input_fn, rec)
            stages["kp0"]["operator_sign_ok"] = ask_while_armed(
                safety, limp, "Was the sign as expected (y/n)?", args, input_fn, rec)
            ask_while_armed(safety, limp, f"Stage 3, low-kp hold: kp {args.hold_kp:g}, kd {args.hold_kd:g} at the "
                            "current position. Press Enter.", args, input_fn, rec)

            start = safety.begin_stage(limits, "hold")
            stages["hold"] = stage_hold(safety, args.hold_time, args.hold_kp, args.hold_kd, args.period, rec)
            stages["hold"]["operator_ok"] = ask_while_armed(
                safety, MitCommand(start, 0.0, args.hold_kp, args.hold_kd, 0.0),
                "Did it hold softly and come back after a push (y/n)?", args, input_fn, rec)
        print("bring-up finished: motor damped and disabled")
    except SafetyFault as e:
        print(f"SAFETY FAULT: {e} (motor damped and disabled)", file=sys.stderr)
        code = 2
    except KeyboardInterrupt:
        print("stopped by Ctrl-C (motor damped and disabled)", file=sys.stderr)
        code = 130
    except SystemExit as e:
        print(f"stopped by signal (motor damped and disabled), exit {e.code}", file=sys.stderr)
        code = e.code if isinstance(e.code, int) else 1
    finally:
        if "stopped" in stages:
            code = 2
        meta.update({
            "start_position": None if safety is None else safety.start_position,
            "stages": stages,
            "result": {"fault": None if safety is None else safety.fault_reason,
                       "disable_acknowledged": None if safety is None else safety.disable_acknowledged,
                       "exit_code": code},
        })
        save_run(run_dir, rec, meta)
        bus.close()
        if fake is not None:
            fake.stop()
        print(f"saved {run_dir}")
    return code


if __name__ == "__main__":
    sys.exit(main())
