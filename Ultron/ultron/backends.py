"""
Ultron AI Backends
Provides a unified interface to:
  - Claude Code (headless `claude -p`, uses your Claude login — no API key)
  - Ollama (local, complete backup when Claude is unreachable)
"""

import os
import re
import json
import time
import shutil
import subprocess
import requests
from typing import Optional

from config import CLAUDE_CODE, OLLAMA, ULTRON_ROOT


# ── Claude Code ──────────────────────────────────────────────────────────────

class ClaudeUnavailable(RuntimeError):
    """Claude Code could not answer (missing binary, logged out, usage limit, timeout)."""

    def __init__(self, message: str, usage_limit: bool = False):
        super().__init__(message)
        self.usage_limit = usage_limit


_USAGE_LIMIT_RE = re.compile(r"usage limit|limit reached|rate.?limit|\b429\b|overloaded|out of (extra )?usage", re.I)

# After a Claude failure we stay on Ollama until this timestamp, then try Claude again.
_claude_retry_after = 0.0


def claude_bin() -> Optional[str]:
    path = CLAUDE_CODE["bin"]
    if os.path.isfile(path) and os.access(path, os.X_OK):
        return path
    return shutil.which("claude")


def claude_code_chat(prompt: str, model: str = None, tools: str = "", cwd: str = None,
                     add_dirs: list = None, timeout: int = None) -> str:
    """One headless Claude Code call. `tools=""` disables all tools (pure text in, text out)."""
    binary = claude_bin()
    if not binary:
        raise ClaudeUnavailable(f"claude binary not found (looked at {CLAUDE_CODE['bin']} and PATH)")

    cmd = [
        binary, "-p",
        "--model", model or CLAUDE_CODE["models"]["heal"],
        "--output-format", "json",
        "--no-session-persistence",
        "--tools", tools,
    ]
    for d in add_dirs or []:
        cmd += ["--add-dir", str(d)]

    # Never bill the API: the CLI must use the subscription login / OAuth token.
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}

    try:
        res = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True, env=env,
            cwd=cwd or str(ULTRON_ROOT), timeout=timeout or CLAUDE_CODE["timeout"],
        )
    except subprocess.TimeoutExpired:
        raise ClaudeUnavailable(f"claude timed out after {timeout or CLAUDE_CODE['timeout']}s")
    except OSError as e:
        raise ClaudeUnavailable(f"could not run claude: {e}")

    data = None
    for line in reversed(res.stdout.strip().splitlines()):
        try:
            data = json.loads(line)
            break
        except ValueError:
            continue

    text = data.get("result", "") if isinstance(data, dict) else ""
    failed = res.returncode != 0 or not isinstance(data, dict) or data.get("is_error")
    if failed or not text:
        detail = (text or res.stderr or res.stdout or "no output").strip()[:400]
        raise ClaudeUnavailable(f"claude failed: {detail}", usage_limit=bool(_USAGE_LIMIT_RE.search(detail)))
    return text


def claude_status() -> tuple[bool, str]:
    """Cheap preflight used by the CLI before submitting."""
    try:
        claude_code_chat("Reply with exactly: OK", model=CLAUDE_CODE["models"]["heal"], timeout=60)
        return True, "ok"
    except ClaudeUnavailable as e:
        return False, str(e)


# ── Ollama ───────────────────────────────────────────────────────────────────

def _ollama_bin() -> Optional[str]:
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        for p in ("/usr/local/bin/ollama", "/usr/bin/ollama", "/opt/ollama/bin/ollama", os.path.expanduser("~/bin/ollama"), os.path.expanduser("~/.local/bin/ollama")):
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
    return ollama_bin


def ensure_ollama_running(host: str = None) -> bool:
    """Checks if Ollama is running; if not, attempts to start it."""
    host = host or OLLAMA["host"]
    try:
        r = requests.get(f"{host}/api/tags", timeout=2)
        if r.status_code == 200:
            return True
    except Exception:
        pass

    ollama_bin = _ollama_bin()
    if ollama_bin:
        try:
            env = os.environ.copy()
            if not env.get("OLLAMA_MODELS") or not os.path.isdir(env.get("OLLAMA_MODELS", "")):
                env["OLLAMA_MODELS"] = os.path.expanduser("~/.ollama/models")
            subprocess.Popen([ollama_bin, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True, env=env)
            for _ in range(8):
                time.sleep(1)
                try:
                    r = requests.get(f"{host}/api/tags", timeout=1)
                    if r.status_code == 200:
                        print("[ULTRON] Ollama server is up and ready.", flush=True)
                        return True
                except Exception:
                    continue
        except Exception as e:
            print(f"[ULTRON] Could not start Ollama daemon: {e}", flush=True)

    return False


def _ensure_ollama_model(host: str, model: str):
    """Pull the backup model if this node doesn't have it yet."""
    try:
        names = [m.get("name", "") for m in requests.get(f"{host}/api/tags", timeout=5).json().get("models", [])]
        if model in names:
            return
        ollama_bin = _ollama_bin()
        if ollama_bin:
            print(f"[ULTRON] Pulling Ollama model {model}...", flush=True)
            subprocess.run([ollama_bin, "pull", model], timeout=1800)
    except Exception as e:
        print(f"[ULTRON] Could not verify/pull Ollama model {model}: {e}", flush=True)


def ollama_chat(prompt: str, model: str = None, host: str = None, timeout: int = None) -> str:
    host    = host    or OLLAMA["host"]
    model   = model   or OLLAMA["model"]
    timeout = timeout or OLLAMA["timeout"]

    if not ensure_ollama_running(host):
        raise RuntimeError(
            f"Ollama not reachable at {host}. "
            "Start it with: `ollama serve` or check OLLAMA_HOST."
        )
    _ensure_ollama_model(host, model)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    try:
        r = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")


def ollama_available(host: str = None) -> bool:
    return ensure_ollama_running(host or OLLAMA["host"])


# ── Unified chat interface ────────────────────────────────────────────────────

BACKENDS = ("claude", "ollama")


def chat(prompt: str, backend: str = "claude", role: str = "heal", **claude_kwargs) -> tuple[str, str]:
    """
    Unified chat call. Returns (reply, model_used).
    backend: 'claude' (Claude Code, falling back to local Ollama) or 'ollama' (local only)
    role:    which CLAUDE_CODE["models"] entry to use: plan | heal | escalate | verify | learn
    """
    global _claude_retry_after
    backend = backend.lower()
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}. Choose from: {list(BACKENDS)}")

    if backend == "claude":
        model = CLAUDE_CODE["models"].get(role, CLAUDE_CODE["models"]["heal"])
        if time.time() >= _claude_retry_after:
            try:
                return claude_code_chat(prompt, model=model, **claude_kwargs), model
            except ClaudeUnavailable as e:
                wait = CLAUDE_CODE["usage_limit_backoff"] if e.usage_limit else CLAUDE_CODE["error_backoff"]
                _claude_retry_after = time.time() + wait
                print(f"[ULTRON] Claude Code unavailable ({e}). Using local Ollama ({OLLAMA['model']}); retrying Claude in {wait}s.", flush=True)

    return ollama_chat(prompt), f"ollama:{OLLAMA['model']}"
