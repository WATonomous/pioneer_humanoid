"""
Ultron Notifier — Discord webhook integration.
Sends rich embeds for job start, steps, errors, and completion.
"""

import os
import json
import time
import datetime
import requests
from typing import Literal, Optional

Status = Literal["started", "step", "error", "done", "warning"]

COLORS = {
    "started": 0x5865F2,   # blurple
    "step":    0x57F287,   # green
    "warning": 0xFEE75C,   # yellow
    "error":   0xED4245,   # red
    "done":    0x23A55A,   # dark green
}

ICONS = {
    "started": "🚀",
    "step":    "⚙️",
    "warning": "⚠️",
    "error":   "❌",
    "done":    "✅",
}


def _webhook_url() -> Optional[str]:
    from config import DISCORD
    if os.environ.get("ULTRON_DISCORD_DISABLE"):   # e.g. for test runs
        return None
    return os.environ.get(DISCORD["webhook_url_env"])


def _ts() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def send(
    status: Status,
    title: str,
    description: str = "",
    fields: Optional[list] = None,
    job_id: str = "",
    task: str = "",
    footer: str = "",
) -> bool:
    """
    Send a Discord embed.  Returns True on success.
    Fields should be list of {"name": ..., "value": ..., "inline": True/False}.
    """
    url = _webhook_url()
    if not url:
        return False  # silently skip if webhook not configured

    icon = ICONS.get(status, "ℹ️")
    embed = {
        "title": f"{icon} {title}",
        "description": description[:4096] if description else "",
        "color": COLORS.get(status, 0x99AAB5),
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "fields": [],
        "footer": {"text": footer or f"Ultron • {_ts()}"},
    }

    if job_id:
        embed["fields"].append({"name": "Job ID", "value": f"`{job_id}`", "inline": True})
    if task:
        embed["fields"].append({"name": "Task", "value": f"```{task[:200]}```", "inline": False})
    if fields:
        embed["fields"].extend(fields)

    payload = {"embeds": [embed]}
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code in (200, 204)
    except Exception:
        return False


def send_step_log(job_id: str, step_number: int, step_name: str, output: str, success: bool = True):
    """Send a detailed step log embed."""
    status: Status = "step" if success else "error"
    title = f"Step {step_number}: {step_name}"
    # Truncate output for Discord (max ~4000 chars in description)
    if len(output) > 3800:
        output = output[:1800] + "\n...[truncated]...\n" + output[-1800:]
    description = f"```\n{output}\n```" if output else "*No output*"
    send(status=status, title=title, description=description, job_id=job_id)


def send_full_log(job_id: str, task: str, log_path: str):
    """Send a final summary with full log file reference."""
    try:
        with open(log_path) as f:
            content = f.read()
    except Exception:
        content = "(log file unavailable)"

    # Discord has 4096 char limit per embed description
    chunks = [content[i:i+3800] for i in range(0, min(len(content), 15000), 3800)]
    for i, chunk in enumerate(chunks):
        send(
            status="done",
            title=f"Full Log (part {i+1}/{len(chunks)})",
            description=f"```\n{chunk}\n```",
            job_id=job_id,
            footer=f"Log: {log_path}",
        )
