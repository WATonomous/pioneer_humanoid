"""
Ultron knowledge — the information/ library.

  information/*.md           hand-written reference guides (highest authority)
  information/learned/*.md   one entry per kind of task, written automatically
                             after a job is verified successful
  information/learned/INDEX.md  one line per learned entry
"""

import os
import re
import fcntl
import datetime
from contextlib import contextmanager

from config import INFORMATION_DIR, LEARNED_DIR

INDEX_FILE = LEARNED_DIR / "INDEX.md"


# ── Secret scrubbing ─────────────────────────────────────────────────────────

_SECRET_PATTERNS = [
    (r'hf_[a-zA-Z0-9]{20,}', '[REDACTED_HF_TOKEN]'),
    (r'gsk_[a-zA-Z0-9]{20,}', '[REDACTED_KEY]'),
    (r'sk-(?:ant|or)-[a-zA-Z0-9_\-]{10,}', '[REDACTED_KEY]'),
    (r'https://discord(?:app)?\.com/api/webhooks/[^\s\'"]+', '[REDACTED_WEBHOOK]'),
    (r'(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY|ACCESS_KEY)[A-Z0-9_]*)=(["\']?)(?!\$)[^\s"\']{6,}\2', r'\1=[REDACTED]'),
    (r'(?i)(--(?:token|password|api[-_]key)[= ])\S{6,}', r'\1[REDACTED]'),
]


def sanitize(text: str) -> str:
    """Mask tokens and secrets from logs, Discord messages and learned entries."""
    for pattern, repl in _SECRET_PATTERNS:
        text = re.sub(pattern, repl, text)
    return text


# ── Reading ──────────────────────────────────────────────────────────────────

def curated_context() -> str:
    """Hand-written reference guides, included in full in planning and healing prompts."""
    if not INFORMATION_DIR.exists():
        return ""
    docs = []
    for p in sorted(INFORMATION_DIR.glob("*.md")):
        if p.name == "README.md":
            continue
        try:
            content = p.read_text().strip()
            if content:
                docs.append(f"--- REFERENCE GUIDE: {p.name} ---\n{content}\n")
        except Exception:
            pass
    if not docs:
        return ""
    return "# REPOSITORY & SYSTEM REFERENCE KNOWLEDGE:\n" + "\n".join(docs)


def index_text() -> str:
    try:
        return INDEX_FILE.read_text().strip()
    except OSError:
        return ""


def _entries():
    if not LEARNED_DIR.exists():
        return []
    return [p for p in sorted(LEARNED_DIR.glob("*.md")) if p.name != "INDEX.md"]


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9_\-\.]{4,}", text.lower())}


def relevant_learned(query: str, max_chars: int = 6000, top: int = 3) -> str:
    """Learned entries that overlap with the query (task / failing command / error), for prompts without tools."""
    q = _words(query)
    scored = []
    for p in _entries():
        try:
            content = p.read_text()
        except OSError:
            continue
        score = len(q & _words(content))
        if score >= 3:
            scored.append((score, p.name, content))
    if not scored:
        return ""
    out, used = [], 0
    for _, name, content in sorted(scored, reverse=True)[:top]:
        chunk = content[: max(0, max_chars - used)]
        if not chunk:
            break
        out.append(f"--- LEARNED ENTRY: {name} ---\n{chunk}\n")
        used += len(chunk)
    return (
        "# LEARNED FROM PREVIOUS SUCCESSFUL ULTRON JOBS\n"
        "(lower authority than the reference guides; an entry with failures recorded may be stale)\n" + "\n".join(out)
    )


# ── Writing ──────────────────────────────────────────────────────────────────

def task_key(task: str) -> str:
    """Stable key for 'the same kind of task': durations and numbers are ignored."""
    t = task.lower()
    t = re.sub(r'\b(for\s+)?\d+\s*(hours?|hrs?|h|minutes?|mins?|m|gpus?)\b', ' ', t)
    t = re.sub(r'[^a-z0-9 ]', ' ', t)
    words = [w for w in t.split() if not w.isdigit()][:10]
    return ("_".join(words) or "task")[:80]


@contextmanager
def _lock():
    LEARNED_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEARNED_DIR / ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _atomic_write(path, text: str):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _read_meta(path) -> dict:
    meta = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return meta
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                break
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta


def _set_meta(text: str, updates: dict) -> str:
    lines = text.splitlines()
    end = lines.index("---", 1)
    seen = set()
    for i in range(1, end):
        k = lines[i].partition(":")[0].strip()
        if k in updates:
            lines[i] = f"{k}: {updates[k]}"
            seen.add(k)
    extra = [f"{k}: {v}" for k, v in updates.items() if k not in seen]
    return "\n".join(lines[:end] + extra + lines[end:]) + "\n"


def _rebuild_index():
    lines = [
        "# Learned entries",
        "",
        "Written automatically by Ultron after verified-successful jobs. One line per entry.",
        "",
    ]
    for p in _entries():
        m = _read_meta(p)
        note = f"verified {m.get('last_verified', '?')}, {m.get('successes', '1')} success(es)"
        if m.get("failures_since", "0") not in ("0", ""):
            note += f", {m['failures_since']} FAILED run(s) since — may be stale"
        lines.append(f"- [{p.name}]({p.name}) — {m.get('task', p.stem)[:160]} ({note})")
    _atomic_write(INDEX_FILE, "\n".join(lines) + "\n")


def _code(cmd: str, lang: str = "bash") -> str:
    return f"```{lang}\n{cmd.strip()}\n```"


def record_success(task: str, ctx: dict, steps: list, fixes: list, duration_s: float, lessons: str = "") -> str:
    """Write/update the learned entry for this task. Returns the entry path."""
    key = task_key(task)
    path = LEARNED_DIR / f"{key}.md"
    today = datetime.date.today().isoformat()

    with _lock():
        successes = int(_read_meta(path).get("successes", "0") or 0) + 1
        res = ctx.get("resources") or {}
        body = [
            "---",
            f"key: {key}",
            f"task: {' '.join(task.split())}",
            f"successes: {successes}",
            "failures_since: 0",
            f"last_verified: {today}",
            f"mode: {ctx.get('mode', '?')}",
            f"node: {ctx.get('node', '?')}",
            "---",
            f"# {' '.join(task.split())}",
            "",
            "## Run details",
            f"- Mode: {ctx.get('mode', '?')} on node {ctx.get('node', '?')}",
            f"- Resources: " + (", ".join(f"{k}={v}" for k, v in res.items()) or "defaults"),
            f"- Duration: {int(duration_s // 60)} min",
            f"- Working directory: {ctx.get('workdir', '?')}",
            "",
            "## Working commands",
            "The sequence that succeeded, in order (failed attempts are left out).",
            "",
        ]
        n = 0
        for s in steps:
            if not s.get("success") or not s.get("cmd") or s.get("superseded"):
                continue
            n += 1
            body += [f"{n}. **{s.get('name', 'step')}**", _code(s["cmd"], s.get("lang", "bash")), ""]

        if fixes:
            body += ["## Problems hit and the fixes that worked", ""]
            for f in fixes:
                body += [
                    f"### {f.get('step', 'step')}",
                    f"Failed command:",
                    _code(f.get("failed_cmd", "")),
                    f"Error (tail):",
                    _code(f.get("error", "")[-800:], "text"),
                    f"Fix (after {f.get('attempts', '?')} attempt(s), by {f.get('model', '?')}):",
                    _code("\n".join(f.get("fix_cmds", []))),
                    "",
                ]
        if lessons.strip():
            body += ["## Lessons", lessons.strip(), ""]

        _atomic_write(path, sanitize("\n".join(body)))
        _rebuild_index()
    return str(path)


def note_failure(task: str):
    """A later run of the same kind of task failed verification: flag the entry as possibly stale."""
    path = LEARNED_DIR / f"{task_key(task)}.md"
    if not path.exists():
        return
    with _lock():
        try:
            failures = int(_read_meta(path).get("failures_since", "0") or 0) + 1
            _atomic_write(path, _set_meta(path.read_text(), {"failures_since": failures}))
            _rebuild_index()
        except Exception:
            pass
