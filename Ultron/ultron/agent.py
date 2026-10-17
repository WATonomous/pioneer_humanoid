"""
Ultron Agent — runs inside the SLURM job (sbatch allocation or srun --overlap step).

This is what executes on the compute node:
1. Loads the job spec and the plan Claude Code wrote at submit time
2. Executes each step, logging everything
3. Self-heals failed steps with no attempt limit (Haiku first, Sonnet on escalation)
4. Verifies the goal, and records what worked in information/learned/
5. Sends Discord notifications at every stage

Usage (internal, called from job script):
    python agent.py --spec /path/to/ultron/jobs/<job_name>.json --job-id 12345
"""

import os
import sys
import re
import json
import time
import signal
import argparse
import subprocess
import datetime
import traceback
from typing import Optional
from pathlib import Path

# Add ultron dir to path so we can import our modules
ULTRON_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ULTRON_DIR))

import envfile
envfile.load_env()

import config
import notifier
import backends
import planning
import knowledge
import safety
from config import HEALING


# ── Logger ────────────────────────────────────────────────────────────────────

class AuditLogger:
    """Writes every event to a structured log file and stdout."""

    def __init__(self, log_path: str, job_id: str, task: str):
        self.log_path = Path(log_path)
        self.job_id   = job_id
        self.task     = task
        self.steps    = []
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_header()

    def _ts(self):
        return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    def _write_header(self):
        header = (
            f"{'='*70}\n"
            f"ULTRON JOB LOG\n"
            f"Job ID : {self.job_id}\n"
            f"Task   : {knowledge.sanitize(self.task)}\n"
            f"Started: {self._ts()}\n"
            f"{'='*70}\n\n"
        )
        self.log_path.write_text(header)
        print(header, flush=True)

    def log(self, level: str, message: str):
        message = knowledge.sanitize(message)
        ts    = self._ts()
        line  = f"[{ts}] [{level.upper():7s}] {message}\n"
        print(line, end="", flush=True)
        with open(self.log_path, "a") as f:
            f.write(line)

    def log_step(self, step_num, name: str, output: str = "", success: bool = True,
                 send_discord: bool = True, cmd: str = "", lang: str = "bash"):
        status = "OK" if success else "FAIL"
        self.log("STEP", f"#{step_num} [{status}] {name}")
        if output:
            for line in output.splitlines():
                self.log("OUT", f"  {line}")
        self.steps.append({
            "step": step_num,
            "name": name,
            "success": success,
            "cmd": cmd,
            "lang": lang,
            "output": output[-2000:],
            "ts": self._ts(),
        })
        if send_discord:
            notifier.send_step_log(
                job_id=self.job_id,
                step_number=step_num,
                step_name=name,
                output=knowledge.sanitize(output),
                success=success,
            )

    def finalize(self, success: bool, summary: str = ""):
        self.log("DONE" if success else "FAIL", f"Job finished. {'SUCCESS' if success else 'FAILED'}")
        if summary:
            self.log("SUMM", summary)
        footer = (
            f"\n{'='*70}\n"
            f"RESULT : {'SUCCESS' if success else 'FAILED'}\n"
            f"Ended  : {self._ts()}\n"
            f"Log    : {self.log_path}\n"
            f"{'='*70}\n"
        )
        with open(self.log_path, "a") as f:
            f.write(footer)
        print(footer, flush=True)


class RunContext:
    """Everything the executor and healer need about this run."""

    def __init__(self, spec: dict, job_id: str, logger: AuditLogger):
        self.spec    = spec
        self.task    = spec["task"]
        self.backend = spec.get("backend", "claude")
        self.workdir = spec.get("workdir") or os.getcwd()
        self.mock    = bool(spec.get("mock"))
        self.job_id  = job_id
        self.logger  = logger
        self.fixes   = []          # healed failures, recorded into information/learned/
        self.started = time.time()

    def run_info(self) -> dict:
        info = dict(self.spec.get("context") or {})
        info.setdefault("mode", self.spec.get("mode", "new"))
        info["node"] = os.uname().nodename
        info["workdir"] = self.workdir
        return info


# ── Shell execution ───────────────────────────────────────────────────────────

def run_shell(cmd: str, workdir: str = None, timeout: int = None) -> tuple[str, int]:
    """Run a shell command, return (combined_output, returncode). No timeout unless the step sets one."""
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=workdir,
        text=True,
        executable="/bin/bash",   # plans use bash features (pipefail, [[ ]], arrays)
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        out += "\n[KILLED: timeout exceeded]"
    return out, proc.returncode


def run_guarded(cmd: str, ctx: RunContext, timeout: int = None, scan: str = None) -> tuple[str, int]:
    """run_shell behind the destructive-command blocklist."""
    reason = safety.check_blocked(scan if scan is not None else cmd, ctx.workdir)
    if reason:
        ctx.logger.log("BLOCK", f"Refused by safety policy ({reason}): {cmd}")
        return (
            f"[BLOCKED by Ultron safety policy: {reason}] This command was NOT run and will never be allowed. "
            f"Achieve the objective a different, non-destructive way."
        ), 126
    return run_shell(cmd, workdir=ctx.workdir, timeout=timeout)


def step_command(step: dict) -> str:
    return step.get("code", "") if step.get("type") == "python" else step.get("cmd", "")


def run_step(step: dict, ctx: RunContext, tag: str) -> tuple[str, int, bool]:
    """Runs one shell/python step plus its optional success check. Returns (output, rc, success)."""
    timeout = step.get("timeout")
    if step.get("type") == "python":
        ctx.logger.log("RUN", "Python snippet")
        tmp = f"/tmp/ultron_step_{os.getpid()}_{tag}.py"
        with open(tmp, "w") as f:
            f.write(step.get("code", ""))
        out, rc = run_guarded(f"python3 {tmp}", ctx, timeout=timeout, scan=step.get("code", ""))
    else:
        cmd = step.get("cmd", "")
        ctx.logger.log("RUN", f"Shell: {cmd}")
        out, rc = run_guarded(cmd, ctx, timeout=timeout)

    # Check semantic errors in output (e.g. wrapper returned 0 on failure)
    semantic_err = check_output_for_failure(out)
    if semantic_err and rc == 0:
        ctx.logger.log("WARN", f"Step returned code 0 but output reported error: {semantic_err}")
        rc = 1
    success = (rc == 0)

    check = step.get("check")
    if success and check:
        cout, crc = run_guarded(check, ctx, timeout=600)
        if crc != 0:
            ctx.logger.log("WARN", f"Step exited 0 but its success check failed: {check}")
            out += f"\n[SUCCESS CHECK FAILED (exit {crc})] $ {check}\n{cout}"
            rc, success = crc, False
    return out, rc, success


# ── Plan generation (in-job fallback; normally the CLI plans at submit time) ──

def generate_plan(ctx: RunContext) -> list[dict]:
    logger = ctx.logger
    if ctx.mock:
        logger.log("INFO", "Running in MOCK mode — skipping LLM plan generation...")
        return [
            {"type": "note",   "name": "Task Received", "message": f"Processing task: {ctx.task}"},
            {"type": "shell",  "name": "System & Python Check", "cmd": "python3 --version && hostname"},
            {"type": "shell",  "name": "Check GPU Availability", "cmd": "nvidia-smi || echo 'No GPU active on this node'"},
            {"type": "note",   "name": "Test Complete", "message": "All mock execution steps verified successfully."}
        ]

    plan_file = ctx.spec.get("plan_file")
    if plan_file and Path(plan_file).is_file():
        plan = json.loads(Path(plan_file).read_text())
        steps = plan.get("steps", []) if isinstance(plan, dict) else plan
        if steps:
            logger.log("INFO", f"Loaded submit-time plan ({len(steps)} steps, by {plan.get('model', '?') if isinstance(plan, dict) else '?'}): {plan_file}")
            return steps
        logger.log("WARN", f"Plan file {plan_file} has no steps. Planning in-job instead.")

    base_prompt = planning.build_plan_prompt(ctx.task, ctx.run_info(), with_tools=False)
    logger.log("INFO", f"No submit-time plan. Generating plan in-job via backend={ctx.backend}...")
    last_err = None
    for attempt in range(1, 4):
        prompt = base_prompt
        if last_err:
            logger.log("WARN", f"Retrying plan generation (attempt {attempt}/3) after: {last_err}")
            prompt += (
                f"\n\nATTENTION: Previous response failed to parse as valid JSON ({last_err}). "
                f"Output strictly the JSON object with no markdown fences, no trailing commas, and properly escaped strings."
            )
        try:
            raw, model = backends.chat(prompt, backend=ctx.backend, role="plan", timeout=config.CLAUDE_CODE["plan_timeout"])
            steps, _ = planning.parse_plan(raw)
            logger.log("INFO", f"Plan generated by {model}: {len(steps)} steps")
            return steps
        except Exception as e:
            last_err = e
            logger.log("WARN", f"Plan generation attempt {attempt} failed: {e}")
    raise ValueError(f"Failed to generate a valid execution plan after 3 attempts. Last error: {last_err}")


# ── Autonomous Debugging & Self-Healing ───────────────────────────────────────

def _render_history(history: list) -> str:
    if not history:
        return "(none yet — this is the first recovery attempt)"
    keep = HEALING["history_full"]
    lines = []
    for h in history[:-keep]:
        cmds = " && ".join(h["cmds"])[:160] or h.get("note", "")
        last = (h["output"].strip().splitlines() or [""])[-1][:160]
        lines.append(f"Attempt {h['n']} [{h['model']}]: {cmds} -> FAILED: {last}")
    for h in history[-keep:]:
        lines.append(f"Attempt {h['n']} [{h['model']}] -> FAILED")
        if h.get("note"):
            lines.append(f"  Note: {h['note']}")
        for c in h["cmds"]:
            lines.append(f"  $ {c}")
        if h["output"]:
            lines.append("  Output (tail):\n    " + h["output"][-1500:].replace("\n", "\n    "))
    return "\n".join(lines)


def _plan_progress(logger: AuditLogger) -> str:
    """Earlier steps with their full commands, so the healer can see what was built and how it is used."""
    cap = HEALING["progress_cmd_chars"]
    rows = []
    for s in logger.steps[-25:]:
        if s.get("superseded"):
            continue
        row = f"- {s['name']}: {'OK' if s['success'] else 'FAILED'}"
        if s.get("cmd"):
            row += "\n    $ " + s["cmd"][:cap].replace("\n", "\n      ")
        rows.append(row)
    text = "\n".join(rows)
    return text[-HEALING["progress_chars"]:] or "(no steps completed yet)"


_PATH_RE = re.compile(r"(?<![\w$])(/[\w.\-+@%/]+)")


def _referenced_files(cmd: str) -> str:
    """Inlines small local files named in the failing command, so the healer sees what actually runs."""
    cap, seen, blocks = HEALING["ref_file_chars"], set(), []
    for path in _PATH_RE.findall(cmd or ""):
        if path in seen or path.startswith(("/proc", "/sys", "/dev")):
            continue
        seen.add(path)
        try:
            p = Path(path)
            if p.is_file() and p.stat().st_size < 1_000_000:
                blocks.append(f"--- {path} ---\n{p.read_text(errors='replace')[:cap]}")
        except OSError:
            continue
    return "\n".join(blocks[:4])


_ERR_LINE = re.compile(r"error|exception|traceback|failed|not found|no such file|denied|cannot|can't", re.I)


def error_signature(text: str) -> str:
    """Normalised terminal error line; equal signatures before and after a fix mean the fix had no effect."""
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    core = ([l for l in lines if _ERR_LINE.search(l)] or lines or [""])[-1]
    return re.sub(r"0x[0-9a-f]+|\d+", "#", core.lower())[:200]


def build_heal_prompt(ctx: RunContext, step: dict, cmd: str, exit_code: int, error_output: str,
                      history: list, remaining: list, tier: str, allow_replan: bool, stalls: int = 0) -> str:
    name = step.get("name", "Unknown step")
    ref_files = _referenced_files(f"{step_command(step)}\n{cmd}")
    ref_text = f"\nLocal files referenced by the failing command:\n{ref_files}\n" if ref_files else ""
    stall_text = ""
    if stalls:
        stall_text = (
            f"\n!! The terminal error is IDENTICAL to the one before your last {stalls} fix(es): those fixes had NO "
            f"effect, so the assumed cause is wrong or the wrong thing was changed. Do not propose another variation "
            f"of them. Find, in the plan progress above, where the failing command actually obtains the thing that is "
            f"missing or wrong (which interpreter, file, path or environment it really uses) and change exactly that.\n"
        )
    learned = knowledge.relevant_learned(f"{ctx.task}\n{cmd}\n{error_output[-1500:]}")
    control = []
    if tier == "heal":
        control.append(
            'If this failure is too complicated for you to resolve with confidence, output exactly\n'
            '   [{"type": "escalate", "reason": "..."}]\n   and a stronger model will take over.'
        )
    if allow_replan:
        control.append(
            'Patching this step has failed many times. You may instead REWRITE THE REST OF THE PLAN: output\n'
            '   [{"type": "replan", "reason": "..."}, <step>, <step>, ...]\n'
            '   where the steps replace the failed step AND all remaining steps below, and still achieve the overall goal.\n'
            f'   Remaining steps currently planned:\n{json.dumps(remaining, indent=2)[:4000]}'
        )
    control_text = "\n".join(f"{i}. {c}" for i, c in enumerate(control, 4))

    return f"""You are Ultron in Autonomous Debugging Mode. A command failed during execution.
Diagnose the error and return corrective recovery steps as a JSON array.

Overall Goal: {ctx.task}

{planning.run_context_text(ctx.run_info())}

Plan progress so far:
{_plan_progress(ctx.logger)}

Failed Step: {name}
Original Command: {step_command(step)}
Success check for this step: {step.get('check') or '(none)'}
Most recent command run: {cmd}
Exit Code: {exit_code}
Terminal Output & Error (tail):
```
{error_output[-3000:]}
```
{ref_text}{stall_text}
Previous recovery attempts on this step (ALL FAILED — do not repeat any of them; an identical proposal is rejected):
{_render_history(history)}

{planning.SAFETY_NOTICE}
{knowledge.curated_context()}

{learned}

Instructions:
1. Diagnose the ROOT cause (e.g., invalid CLI flag, wrong path, missing dependency, daemon not ready). Use the
   attempt history: if earlier fixes failed, your diagnosis must differ from theirs.
2. Generate 1 to 4 concrete recovery steps that fix the problem AND accomplish this step's original objective
   (the last step is normally the corrected version of the original command).
3. Output ONLY a valid JSON array of steps:
   [{{"type": "shell", "name": "...", "cmd": "...", "why": "...", "check": "..."}}, ...]
   - "why" (first step): one sentence naming your root-cause hypothesis and the evidence for it.
   - "check": every step EXCEPT the last must have one: a shell command that exits 0 only if that step's intended
     effect is really in place (e.g. grep the edited file for the new value, or import the module with the exact
     interpreter the failing command uses). A command that exits 0 is not proof that it changed anything. Proposals
     missing these checks are rejected unrun. The last step is the corrected original command.
{control_text}
No markdown fences or text outside JSON.
"""


def attempt_healing(step: dict, error_output: str, exit_code: int, remaining: list, ctx: RunContext) -> Optional[list]:
    """
    Diagnoses a failed step via LLM and executes corrective steps until it works.
    There is no attempt limit: the loop ends on success, SLURM wall time, or `ultron --cancel`.
    Returns None when the step was healed, or a list of steps that replaces the remaining plan.
    """
    logger = ctx.logger
    name = step.get("name", "Unknown step")
    failed_cmd = step_command(step)
    first_error = error_output
    cmd = failed_cmd
    history, seen = [], set()
    tier = "heal"
    attempt = 0
    last_sig, stalls = error_signature(first_error), 0

    def escalate(reason: str):
        nonlocal tier
        if tier == "heal":
            tier = "escalate"
            logger.log("WARN", f"Escalating '{name}' from {config.CLAUDE_CODE['models']['heal']} to {config.CLAUDE_CODE['models']['escalate']}: {reason}")

    def record(model: str, cmds: list, output: str, note: str = ""):
        history.append({"n": attempt, "model": model, "cmds": cmds, "output": output, "note": note})

    while True:
        attempt += 1
        if attempt > HEALING["haiku_attempts"]:
            escalate(f"{HEALING['haiku_attempts']} attempts failed")
        allow_replan = (tier == "escalate" and bool(remaining)
                        and (attempt > HEALING["replan_after"] or stalls >= HEALING["stall_replan"]))

        if attempt % HEALING["notify_every"] == 0:
            notifier.send(
                status="warning",
                title=f"Still self-healing: attempt {attempt}",
                description=f"Step **{name}** has not recovered yet. Run `ultron --cancel {ctx.job_id}` to stop.\n"
                            f"```\n{knowledge.sanitize(error_output[-1200:])}\n```",
                job_id=ctx.job_id,
            )

        logger.log("DEBUG", f"Self-healing attempt {attempt} for '{name}' (tier: {tier})...")
        prompt = build_heal_prompt(ctx, step, cmd, exit_code, error_output, history, remaining, tier, allow_replan, stalls)
        try:
            raw, model = backends.chat(prompt, backend=ctx.backend, role=tier)
        except Exception as e:
            wait = HEALING["all_backends_down_wait"]
            logger.log("WARN", f"No model reachable for healing ({e}). Waiting {wait}s and trying again...")
            attempt -= 1
            time.sleep(wait)
            continue

        try:
            recovery_steps = [s for s in planning.parse_plan_json(raw) if isinstance(s, dict)]
        except Exception as e:
            logger.log("DEBUG", f"Healing reply from {model} was not valid JSON: {e}")
            record(model, [], "", note="reply was not a valid JSON array of steps")
            escalate("unparseable reply")
            continue

        head = recovery_steps[0] if recovery_steps else {}
        if head.get("type") == "escalate":
            record(model, [], "", note=f"model declined: {head.get('reason', '')}")
            escalate(f"{model} reports the problem is beyond it ({head.get('reason', 'no reason given')})")
            continue
        if head.get("type") == "replan":
            new_steps = [s for s in recovery_steps[1:] if s.get("type") in ("shell", "python", "note")]
            if allow_replan and any(s.get("type") != "note" for s in new_steps):
                logger.log("WARN", f"Re-planning from '{name}' onward ({len(new_steps)} new steps): {head.get('reason', '')}")
                logger.log("PLAN", f"\n{json.dumps(new_steps, indent=2)}")
                notifier.send(status="warning", title="Plan rewritten by self-healing",
                              description=f"After {attempt} attempts on **{name}**: {head.get('reason', '')}", job_id=ctx.job_id)
                return new_steps
            record(model, [], "", note="proposed a replan that was not allowed or empty")
            continue

        runnable = [s for s in recovery_steps if s.get("type", "shell") in ("shell", "python")]
        signature = tuple(" ".join(step_command(s).split()) for s in runnable)
        if not runnable or signature in seen:
            why = "no executable steps" if not runnable else "identical to an earlier failed attempt"
            logger.log("WARN", f"Recovery proposal from {model} rejected: {why}.")
            record(model, list(signature), "", note=f"REJECTED without running: {why}. Take a different approach.")
            escalate(f"repeated fix ({why})")
            continue
        unchecked = [s.get("name", "?") for s in runnable[:-1] if not s.get("check")]
        if unchecked:
            logger.log("WARN", f"Recovery proposal from {model} rejected: no success check on {unchecked}.")
            record(model, list(signature), "",
                   note=f"REJECTED without running: steps {unchecked} have no \"check\" proving their effect.")
            continue
        seen.add(signature)
        why = next((s["why"] for s in recovery_steps if s.get("why")), "")
        if why:
            logger.log("NOTE", f"Hypothesis ({model}): {why}")

        all_ok = True
        first_record = len(logger.steps)
        for j, rstep in enumerate(recovery_steps, 1):
            rtype = rstep.get("type", "shell")
            rname = rstep.get("name", f"Recovery step {j}")
            if rtype == "note":
                logger.log("NOTE", rstep.get("message", rname))
                continue
            if rtype not in ("shell", "python"):
                continue
            rstep.setdefault("type", "shell")
            logger.log("RUN", f"[RECOVERY #{attempt}.{j}]")
            rout, rrc, rok = run_step(rstep, ctx, tag=f"heal{attempt}_{j}")
            logger.log_step(f"R{attempt}.{j}", f"[RECOVERY] {rname}", output=rout, success=rok,
                            cmd=step_command(rstep), lang="python" if rtype == "python" else "bash")
            if not rok:
                all_ok = False
                cmd, error_output, exit_code = step_command(rstep), rout, rrc
                break

        # The original step's own success check must also pass after recovery.
        if all_ok and step.get("check"):
            cout, crc = run_guarded(step["check"], ctx, timeout=600)
            if crc != 0:
                all_ok = False
                cmd, exit_code = step["check"], crc
                error_output = f"Recovery steps ran, but the step's success check still fails:\n$ {step['check']}\n{cout}"
                logger.log("WARN", f"Recovery ran but success check still fails for '{name}'.")

        if all_ok:
            logger.log("INFO", f"✓ Self-healing successful for '{name}' after {attempt} attempt(s) (by {model}).")
            ctx.fixes.append({
                "step": name, "failed_cmd": failed_cmd, "error": first_error,
                "fix_cmds": list(signature), "attempts": attempt, "model": model,
            })
            return None
        # keep this attempt's steps out of the learned "working commands"
        for rec in logger.steps[first_record:]:
            rec["superseded"] = True
        sig = error_signature(error_output)
        note = ""
        if sig == last_sig:
            stalls += 1
            note = f"ERROR UNCHANGED after this fix (stall #{stalls}): the hypothesis was falsified."
            escalate("error unchanged after fix")
        else:
            stalls = 0
        last_sig = sig
        record(model, list(signature), error_output, note=note or why)


# ── Semantic Failure Detection ───────────────────────────────────────────────

def check_output_for_failure(output: str) -> Optional[str]:
    """Detects fatal error signatures in terminal output even if exit code was 0."""
    patterns = [
        (r"failed to solve", "Docker build solver failed"),
        (r"Error response from daemon", "Docker daemon error"),
        (r"operation not permitted", "Permission denied / operation not permitted"),
        (r"No space left on device", "Disk out of space"),
        (r"container\.py: error: argument command: invalid choice", "Invalid container.py subcommand"),
        (r"RuntimeError: The container '.*' is not running", "Container is not running"),
        (r"RuntimeError: Can't stop container '.*' as it is not running", "Container is not running"),
        (r"Traceback \(most recent call last\)", "Python unhandled exception"),
    ]
    for pat, reason in patterns:
        if re.search(pat, output, re.IGNORECASE):
            return reason
    return None


# ── Autonomous Goal Verification ─────────────────────────────────────────────

def verify_goal_achievement(ctx: RunContext) -> tuple[Optional[bool], str]:
    """
    Inspects all execution outputs to verify if the user's objective was truly accomplished.
    Returns (True/False, reason), or (None, reason) when no model could give a verdict.
    """
    if ctx.mock:
        return True, "Mock verification"

    history = []
    for s in ctx.logger.steps:
        status = "OK" if s.get("success") else "FAIL"
        history.append(f"- Step #{s.get('step')}: {s.get('name')} | Status: {status}")
        lines = [line.strip() for line in s.get("output", "").strip().splitlines() if line.strip()]
        if lines:
            history.append(f"  Output: {' | '.join(lines[-3:])[:300]}")

    prompt = f"""You are the Ultron Quality & Goal Verification Supervisor.
User's Desired Task: "{ctx.task}"

Execution History (steps named [RECOVERY] are self-healing fixes; a FAILED step followed by successful
recovery steps counts as recovered):
{chr(10).join(history)}

Did this run ACTUALLY achieve the user's requested objective?
Note: If a crucial step (such as starting a container, running a test, or finishing training) failed without being recovered, or was never accomplished, the goal was NOT achieved.

Respond ONLY with valid JSON:
{{"achieved": true, "reason": "concise 1-sentence explanation"}}
or
{{"achieved": false, "reason": "concise 1-sentence explanation of what failed"}}
"""
    last_err = "no verdict"
    for role in ("verify", "escalate"):
        try:
            raw, _ = backends.chat(prompt, backend=ctx.backend, role=role)
            clean_raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            m = re.search(r'\{.*\}', clean_raw, re.DOTALL)
            res = json.loads(m.group(0), strict=False)
            if isinstance(res.get("achieved"), bool):
                return res["achieved"], res.get("reason", "Verified by LLM")
            last_err = "verdict missing 'achieved'"
        except Exception as e:
            last_err = str(e)
            ctx.logger.log("WARN", f"Goal verification check error ({role}): {e}")
    return None, f"could not verify ({last_err})"


# ── Learning ─────────────────────────────────────────────────────────────────

def record_knowledge(ctx: RunContext):
    """After a verified success, save what worked to information/learned/ for future Ultron runs."""
    lessons = ""
    if ctx.fixes:
        fixes_text = "\n\n".join(
            f"Step: {f['step']}\nFailed command: {f['failed_cmd']}\nError tail: {f['error'][-600:]}\nFix that worked: {' ; '.join(f['fix_cmds'])}"
            for f in ctx.fixes
        )
        prompt = (
            "These failures were hit and fixed during a successful automated job. Write 2 to 5 short markdown bullets "
            "stating the reusable lessons (root cause -> what to do instead next time). Be specific; no preamble; "
            "never include tokens or secrets.\n\n"
            f"Task: {ctx.task}\n\n{knowledge.sanitize(fixes_text)}"
        )
        try:
            lessons, _ = backends.chat(prompt, backend=ctx.backend, role="learn")
        except Exception as e:
            ctx.logger.log("WARN", f"Could not write lessons section: {e}")

    info = ctx.run_info()
    info["resources"] = ctx.spec.get("resources") or {}
    path = knowledge.record_success(
        task=ctx.task, ctx=info, steps=ctx.logger.steps, fixes=ctx.fixes,
        duration_s=time.time() - ctx.started, lessons=lessons,
    )
    ctx.logger.log("INFO", f"Saved what worked for future runs: {path}")


# ── Plan execution ────────────────────────────────────────────────────────────

def execute_plan(plan: list[dict], ctx: RunContext) -> bool:
    logger = ctx.logger
    overall_success = True
    steps = list(plan)
    i = 0
    while i < len(steps):
        step  = steps[i]
        num   = i + 1
        stype = step.get("type", "note")
        name  = step.get("name", f"Step {num}")

        if stype in ("shell", "python"):
            out, rc, success = run_step(step, ctx, tag=str(num))
            logger.log_step(num, name, output=out, success=success,
                            cmd=step_command(step), lang="python" if stype == "python" else "bash")
            if not success:
                if ctx.mock:
                    overall_success = False
                else:
                    logger.log("WARN", f"Step #{num} failed. Entering autonomous self-healing / debugging...")
                    replacement = attempt_healing(step, out, rc, steps[i + 1:], ctx)
                    if replacement is not None:
                        steps = steps[:i + 1] + replacement

        elif stype == "note":
            msg = step.get("message", step.get("name", ""))
            logger.log_step(num, f"[NOTE] {name}", output=msg, success=True)

        else:
            logger.log("WARN", f"Unknown step type: {stype}, skipping")

        i += 1

    return overall_success


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ultron compute agent")
    parser.add_argument("--spec",   required=True, help="Path to the job spec JSON written by the ultron CLI")
    parser.add_argument("--job-id", required=True, dest="job_id")
    args = parser.parse_args()

    spec = json.loads(Path(args.spec).read_text())
    logger = AuditLogger(log_path=spec["log_path"], job_id=args.job_id, task=spec["task"])
    ctx = RunContext(spec, args.job_id, logger)

    def on_terminate(signum, _frame):
        logger.log("FAIL", f"Received signal {signum} (SLURM wall time reached or job cancelled).")
        logger.finalize(success=False, summary="TERMINATED before completion")
        notifier.send(status="error", title="Ultron Job Terminated",
                      description="Stopped by SLURM (wall time reached or cancelled) before completion.",
                      job_id=args.job_id, task=knowledge.sanitize(ctx.task))
        os._exit(143)

    signal.signal(signal.SIGTERM, on_terminate)

    # Notify: job started
    notifier.send(
        status="started",
        title=f"Ultron Job Started",
        description=knowledge.sanitize(ctx.task),
        job_id=args.job_id,
        fields=[
            {"name": "Backend", "value": "mock" if ctx.mock else ctx.backend, "inline": True},
            {"name": "Mode",    "value": spec.get("mode", "new"), "inline": True},
            {"name": "Node",    "value": os.uname().nodename, "inline": True},
            {"name": "Workdir", "value": ctx.workdir, "inline": False},
        ],
    )

    try:
        plan = generate_plan(ctx)
        plan_json = json.dumps(plan, indent=2)
        logger.log("PLAN", f"\n{plan_json}")
        notifier.send(
            status="step",
            title="Execution Plan",
            description=f"```json\n{knowledge.sanitize(plan_json)[:3500]}\n```",
            job_id=args.job_id,
        )

        success = execute_plan(plan, ctx)

        # Autonomous Goal Verification
        logger.log("INFO", "Verifying whether the overall user objective was actually achieved...")
        achieved, reason = verify_goal_achievement(ctx)
        if achieved is False:
            success = False
            logger.log("FAIL", f"Autonomous Goal Verification Failed: {reason}")
            status_str = f"FAILED (Goal Not Achieved: {reason})"
            knowledge.note_failure(ctx.task)
        elif achieved is None:
            logger.log("WARN", f"Goal could not be verified: {reason}")
            status_str = "UNVERIFIED (steps completed, but no model could confirm the goal)" if success else "PARTIAL FAILURE"
        else:
            logger.log("INFO", f"✓ Autonomous Goal Verification Confirmed: {reason}")
            status_str = "SUCCESS" if success else "PARTIAL FAILURE"
            if success and not ctx.mock:
                try:
                    record_knowledge(ctx)
                except Exception as e:
                    logger.log("WARN", f"Could not save learned entry: {e}")

    except Exception as e:
        tb = traceback.format_exc()
        logger.log("ERROR", f"Fatal error: {e}\n{tb}")
        notifier.send(
            status="error",
            title="Ultron Job FAILED",
            description=f"```\n{knowledge.sanitize(tb)[:3500]}\n```",
            job_id=args.job_id,
        )
        logger.finalize(success=False, summary=str(e))
        sys.exit(1)

    # Finalize
    verified_ok = success and achieved is True
    logger.finalize(success=success, summary=status_str)
    notifier.send(
        status="done" if verified_ok else "warning",
        title=f"Ultron Job {'Complete' if verified_ok else 'Completed with Warnings' if success else 'Completed with Errors'}",
        description=f"Task: {knowledge.sanitize(ctx.task)}\nResult: **{status_str}**",
        job_id=args.job_id,
        fields=[{"name": "Full Log", "value": spec["log_path"], "inline": False}],
    )

    # Send full log to Discord
    notifier.send_full_log(args.job_id, knowledge.sanitize(ctx.task), spec["log_path"])

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
