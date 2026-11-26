#!/usr/bin/env python3
"""Trim old RL training sessions: keep only the most-trained checkpoint.

An "old" session is one dated strictly more than 3 days ago. Today is
2026-10-17, so the cut-off is 2026-10-15: sessions stamped 2026-10-14 or
earlier get every ``model_<iter>.pt`` deleted except the highest iteration.

Everything else in the session (params, git, tfevents, videos, exports) is
left alone. Dated directories only -- undated helpers like
``rough2999_source`` are never touched.

Usage:
    python3 tools/rl_sessions/trim_old_checkpoints.py            # dry run
    python3 tools/rl_sessions/trim_old_checkpoints.py --execute
"""

from __future__ import annotations

import argparse
import os
import re
import sys

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "rl")
_MODEL_RE = re.compile(r"^model_(\d+)\.pt$")

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--root", default=os.path.abspath(_ROOT))
parser.add_argument("--cutoff", default="2026-10-15",
                    help="Keep dated sessions on/after this YYYY-MM-DD (default: 2026-10-15).")
parser.add_argument("--execute", action="store_true", help="Actually delete (default: dry run).")
args_cli = parser.parse_args()


def main() -> int:
    total = 0
    kept_sessions = deleted_sessions = 0
    for task in sorted(os.listdir(args_cli.root)):
        task_dir = os.path.join(args_cli.root, task)
        if not os.path.isdir(task_dir):
            continue
        for session in sorted(os.listdir(task_dir)):
            session_dir = os.path.join(task_dir, session)
            if not os.path.isdir(session_dir):
                continue
            date = re.match(r"(\d{4}-\d{2}-\d{2})_", session)
            if date is None:
                print(f"  keep (undated)      {task}/{session}")
                continue
            if date.group(1) >= args_cli.cutoff:
                kept_sessions += 1
                print(f"  keep (recent)       {task}/{session}")
                continue

            files = [f for f in os.listdir(session_dir) if _MODEL_RE.match(f)]
            if not files:
                print(f"  --                  {task}/{session} (no model_*.pt)")
                continue
            files.sort(key=lambda f: int(_MODEL_RE.match(f).group(1)))
            keep, drop = files[-1], files[:-1]
            freed = sum(os.path.getsize(os.path.join(session_dir, f)) for f in drop)
            total += freed
            deleted_sessions += 1
            print(f"  trim {task}/{session}: keep {keep}, drop {len(drop)} "
                  f"of {len(files)} ({freed / 1e6:.0f} MB)")
            if args_cli.execute:
                for name in drop:
                    os.remove(os.path.join(session_dir, name))

    verb = "freed" if args_cli.execute else "would free"
    print(f"\n{verb} {total / 1e9:.2f} GB across {deleted_sessions} old sessions "
          f"({kept_sessions} recent/undated kept intact)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
