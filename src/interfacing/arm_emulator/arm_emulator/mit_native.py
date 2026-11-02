"""ctypes binding to the real mit_protocol.cpp (native/mit_protocol_c.cpp), built on first use.

    python -m arm_emulator.mit_native     # build it now (the Docker image does this)
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
_CAN = _PKG.parent / "can"
_SOURCES = [_PKG / "native" / "mit_protocol_c.cpp", _CAN / "src" / "mit_protocol.cpp"]
_HEADER = _CAN / "include" / "mit_protocol.hpp"
_LIB = _PKG / "native" / "build" / ("mit_protocol_c.dll" if sys.platform == "win32" else "libmit_protocol_c.so")


def build(force: bool = False) -> Path:
    """Compile the shared library unless it is newer than its sources."""
    newest = max(p.stat().st_mtime for p in [*_SOURCES, _HEADER])
    if force or not _LIB.exists() or _LIB.stat().st_mtime < newest:
        _LIB.parent.mkdir(parents=True, exist_ok=True)
        flags = ["-static"] if sys.platform == "win32" else ["-fPIC"]
        subprocess.run(["g++", "-std=c++17", "-O2", "-shared", *flags, f"-I{_HEADER.parent}", *map(str, _SOURCES),
                        "-o", str(_LIB)], check=True)
    return _LIB


_lib = ctypes.CDLL(str(build()))
_d, _u = ctypes.c_double, ctypes.c_uint
_lib.mit_pack_value.argtypes, _lib.mit_pack_value.restype = [_d, _d, _d, _u], _u
_lib.mit_unpack_value.argtypes, _lib.mit_unpack_value.restype = [_u, _d, _d, _u], _d
_lib.mit_pack_command.argtypes = [ctypes.c_int, *[_d] * 10, ctypes.POINTER(ctypes.c_ubyte)]
_lib.mit_pack_command.restype = None


def pack_value(phys: float, lo: float, hi: float, bits: int) -> int:
    return _lib.mit_pack_value(phys, lo, hi, bits)


def unpack_value(code: int, lo: float, hi: float, bits: int) -> float:
    return _lib.mit_unpack_value(code, lo, hi, bits)


def pack_command(is_ak: bool, p: float, v: float, kp: float, kd: float, t: float, profile: dict) -> bytes:
    """The 8-byte payload can_node would put on the bus for this MotorCmd."""
    out = (ctypes.c_ubyte * 8)()
    _lib.mit_pack_command(int(is_ak), p, v, kp, kd, t, profile["p_max"], profile["v_max"], profile["t_max"],
                          profile["kp_max"], profile["kd_max"], out)
    return bytes(out)


if __name__ == "__main__":
    print(build(force=True))
