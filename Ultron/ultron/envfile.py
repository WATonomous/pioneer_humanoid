"""
Ultron .env loader.
Ultron reads its own .env at startup (CLI and agent), so secrets never need to
be exported in your shell or written into job scripts.
"""

import os
from pathlib import Path


def load_env(path: Path = None) -> dict:
    """Parse KEY=VALUE lines from .env into os.environ. Returns what was loaded."""
    # Resolved here rather than via config: config reads env vars at import time,
    # so .env must be loaded before config is imported.
    path = Path(path) if path else Path(__file__).parent.resolve() / ".env"
    loaded = {}
    if not path.is_file():
        return loaded
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if not key.replace("_", "").isalnum():
            continue
        os.environ[key] = value
        loaded[key] = value
    return loaded
