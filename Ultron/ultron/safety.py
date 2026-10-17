"""
Ultron safety policy — destructive-command blocklist.
The executor calls check_blocked() before running any plan or recovery command.
"""

import os
import re
import shlex
from typing import Optional

from config import SAFETY


def _rm_reason(cmd: str, workdir: str) -> Optional[str]:
    """Recursive deletes are only allowed on relative paths inside the workdir or under the safe prefixes."""
    wd = os.path.realpath(workdir) if workdir else None

    def allowed(path: str) -> bool:
        if wd and (path == wd or path.startswith(wd.rstrip("/") + "/")):
            return True
        return any(path.startswith(prefix) for prefix in SAFETY["rm_safe_prefixes"])

    for m in re.finditer(r'\brm\s+([^;&|\n]*)', cmd):
        try:
            args = shlex.split(m.group(1))
        except ValueError:
            args = m.group(1).split()
        flags = [a for a in args if a.startswith("-")]
        recursive = any(a == "--recursive" or (not a.startswith("--") and re.search(r"[rR]", a)) for a in flags)
        if not recursive:
            continue
        for target in (a for a in args if not a.startswith("-")):
            t = target.strip("'\"")
            if t in ("*", ".", "./", "./*", "/", "/*"):
                return f"recursive delete of '{t}'"
            if t.startswith(("~", "$")) or ".." in t.split("/"):
                return f"recursive delete of '{t}' (cannot be verified to stay inside the working directory)"
            if t.startswith("/"):
                if not allowed(os.path.normpath(t)):
                    return f"recursive delete outside the working directory ('{t}')"
    return None


def check_blocked(cmd: str, workdir: str = None) -> Optional[str]:
    """Returns the reason a command is refused, or None if it may run."""
    for pattern, reason in SAFETY["blocked_patterns"]:
        if re.search(pattern, cmd):
            return reason
    return _rm_reason(cmd, workdir)
