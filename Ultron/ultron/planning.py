"""
Ultron planning — prompts and plan parsing shared by the CLI (submit-time
planning with Claude Code) and the agent (in-job fallback planning, healing).
"""

import re
import json

import knowledge


SYSTEM_PROMPT = """You are Ultron, an autonomous AI agent that runs tasks inside SLURM compute jobs.
Your job is to take a high-level task description and produce a concrete, detailed execution plan.

Output a single JSON object:
{
  "resources": {"time": "H:MM:SS", "gpus": 1, "cpus": 8, "mem": "48G", "tmpdisk_mb": 0},
  "steps": [ ... ]
}

Each step must be one of:
  {"type": "shell", "name": "...", "cmd": "...", "check": "..."}   <- run a shell command
  {"type": "python", "name": "...", "code": "..."}                 <- run python3 code inline
  {"type": "note",   "name": "...", "message": "..."}              <- log a note (no execution)

"check" is optional: a quick shell command that exits 0 only if the step really achieved its purpose
(e.g. `docker ps --format '{{.Names}}' | grep -qx isaac-lab-ros2`, `test -f outputs/ckpt/last.pt`).
Add one wherever a command could exit 0 without having worked.

Rules:
- Output ONLY the JSON object. No markdown fences. No explanation outside JSON.
- "resources" is what the whole plan needs: wall time (include setup + margin), GPUs, CPUs, memory, and
  tmpdisk_mb (node-local scratch in MB; request 102400 when the plan runs dockerd or builds images, else 0).
- Steps must be concrete and executable, in order, each doing one thing.
- For ML training tasks, include: environment setup, config validation, actual training command.
- Always include a final "note" step summarizing what was done.
- If the task is ambiguous, make sensible assumptions and log them as notes.
- Keep step names short (< 60 chars).
- Keep inline Python code steps concise. For large files (like READMEs), write them via shell commands (e.g. cat << 'EOF' > README.md) or compact scripts rather than embedding giant multi-line documents inside JSON string properties.
- Grounding & Discovery First: Do not guess nonexistent filenames (e.g. train_vla.py) or guess CLI flags. If inspecting a task directory or unfamiliar repository, check existing entrypoints or README.md first.
- Remote Datasets vs Local Paths: Hugging Face dataset IDs (formatted as 'Username/DatasetName', e.g. CursedRock17/...) are remote repository identifiers for libraries/tools, not local folders on disk. Do not test them with 'ls /workspace/...'.
- Package Imports: Always set PYTHONPATH to include relevant repository source directories (e.g. PYTHONPATH=/workspace/humanoid/src:/workspace/humanoid/src/il:$(pwd)) when invoking python scripts so internal package imports resolve.
- Secrets: never write tokens or keys into commands. They are already in the job environment (e.g. $HF_TOKEN).
- Autonomous Debugging Mindset: Formulate concrete commands. If a command fails during execution, an autonomous healing loop will analyze the error output, diagnose the issue, and formulate corrective steps to recover and finish the task.
"""

SAFETY_NOTICE = """Safety policy — the executor REFUSES these commands, so never plan them:
recursive `rm` outside the job working directory, `docker ... prune` / `docker volume rm`, disk formatting or raw
device writes, `scancel`, git force pushes, anything touching `.claude/` or modifying a `.env` file, powering off the node.
"""


def run_context_text(ctx: dict) -> str:
    """Describes where the job will run, so plans fit the allocation."""
    mode = ctx.get("mode", "new")
    lines = ["# RUN CONTEXT", f"Working directory: {ctx.get('workdir', '')}"]
    if mode == "overlap":
        lines += [
            f"Mode: OVERLAP — runs as an extra step inside the already-running SLURM job {ctx.get('parent_job', '?')} "
            f"on node {ctx.get('node', '?')}.",
            f"Allocation: {ctx.get('gres') or 'unknown GRES'}; time left on the allocation: {ctx.get('time_left', '?')}.",
            "The node already carries state from that job: dockerd and containers (e.g. isaac-lab-ros2) may already be "
            "running. CHECK for them and REUSE them; do not rebuild images or restart the daemon unless a check shows "
            "they are missing. The plan's wall time must fit inside the time left.",
        ]
    else:
        lines += [
            "Mode: NEW — a fresh sbatch allocation, possibly on a different node than previous runs.",
            f"Partition: {ctx.get('partition', '?')}"
            + (f"; requested node: {ctx['node']}" if ctx.get("node") else "")
            + (f"; GPU type: {ctx['gpu_type']}" if ctx.get("gpu_type") else ""),
            "Nothing is running on the node: no dockerd, no containers, and node-local Docker images may be absent. "
            "The plan must start dockerd and build/pull/start whatever container it needs before using it.",
        ]
    return "\n".join(lines)


def build_plan_prompt(task: str, ctx: dict, with_tools: bool) -> str:
    parts = [SYSTEM_PROMPT, SAFETY_NOTICE, run_context_text(ctx), knowledge.curated_context()]
    index = knowledge.index_text()
    if with_tools:
        parts.append(
            "# HOW TO GROUND THE PLAN\n"
            "You have read-only tools (Read, Glob, Grep). Before writing the plan, inspect the working directory "
            "to confirm real script paths, entrypoints and CLI flags instead of guessing."
        )
        if index:
            parts.append(
                f"# LEARNED FROM PREVIOUS SUCCESSFUL ULTRON JOBS\n"
                f"Entries live in {knowledge.LEARNED_DIR}. Read the ones relevant to this task and reuse their "
                f"working commands and fixes. They are lower authority than the reference guides above and than "
                f"what you see in the repository; an entry marked with failures may be stale.\n{index}"
            )
    else:
        learned = knowledge.relevant_learned(task)
        if learned:
            parts.append(learned)
    parts.append(f"Task: {task}\n\nProduce the execution plan JSON object:")
    return "\n\n".join(p for p in parts if p)


def parse_plan_json(raw: str) -> list[dict]:
    """Robustly extracts and parses a JSON step array from raw LLM output,
    handling markdown fences, reasoning tags, raw newlines, trailing commas,
    and truncated outputs."""
    clean_raw = raw.strip()
    clean_raw = re.sub(r'<think>.*?</think>', '', clean_raw, flags=re.DOTALL).strip()

    # 1. Direct parse if whole string is JSON array
    if clean_raw.startswith('[') and clean_raw.endswith(']'):
        try:
            p = json.loads(clean_raw, strict=False)
            if isinstance(p, list):
                return p
        except Exception:
            pass

    # 2. Markdown fence extraction
    fence_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', clean_raw, re.DOTALL)
    if fence_match:
        try:
            p = json.loads(fence_match.group(1), strict=False)
            if isinstance(p, list):
                return p
        except Exception:
            pass

    # 3. Balanced bracket scanning (skipping brackets inside quotes)
    start = clean_raw.find('[')
    if start != -1:
        s = clean_raw[start:]
        in_string = False
        escape = False
        depth = 0
        end_idx = -1
        stack = []

        for i, c in enumerate(s):
            if escape:
                escape = False
            elif c == '\\':
                escape = True
            elif c == '"':
                in_string = not in_string
            elif not in_string:
                if c in '[{':
                    stack.append(c)
                    if c == '[':
                        depth += 1
                elif c in ']}':
                    if stack:
                        stack.pop()
                    if c == ']':
                        depth -= 1
                        if depth == 0:
                            end_idx = i
                            break

        if end_idx != -1:
            candidate = s[:end_idx + 1]
            try:
                p = json.loads(candidate, strict=False)
                if isinstance(p, list):
                    return p
            except Exception:
                # Remove trailing commas before ] or }
                candidate_fixed = re.sub(r',\s*([\]}])', r'\1', candidate)
                try:
                    p = json.loads(candidate_fixed, strict=False)
                    if isinstance(p, list):
                        return p
                except Exception:
                    pass

        # 4. Truncation repair: close open strings and open structures if output was cut off
        repaired = s
        if in_string:
            repaired += '"'
        for open_char in reversed(stack):
            if open_char == '{':
                repaired += '}'
            elif open_char == '[':
                repaired += ']'
        repaired = re.sub(r',\s*([\]}])', r'\1', repaired)
        try:
            p = json.loads(repaired, strict=False)
            if isinstance(p, list):
                return p
        except Exception:
            pass

    # 5. Greedy regex fallback
    m = re.search(r'\[.*\]', clean_raw, re.DOTALL)
    if m:
        try:
            p = json.loads(m.group(0), strict=False)
            if isinstance(p, list):
                return p
        except Exception:
            pass

    # 6. Salvage completed steps if only the trailing step was truncated / malformed
    idx = clean_raw.rfind("},")
    while idx != -1:
        candidate = clean_raw[:idx+1] + "\n]"
        c_start = candidate.find("[")
        if c_start != -1:
            try:
                p = json.loads(candidate[c_start:], strict=False)
                if isinstance(p, list) and len(p) > 0:
                    return p
            except Exception:
                pass
        idx = clean_raw.rfind("},", 0, idx)

    raise ValueError(f"No valid JSON array found in AI response (length {len(raw)}): {raw[:300]}...")


def parse_plan(raw: str) -> tuple[list, dict]:
    """Parses a planner reply into (steps, resources). Accepts the object form or a bare step array."""
    clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', clean)
    start = clean.find('{')
    array_start = clean.find('[')
    if start != -1 and (array_start == -1 or start < array_start):
        try:
            obj, _ = json.JSONDecoder(strict=False).raw_decode(clean[start:])
            if isinstance(obj, dict) and isinstance(obj.get("steps"), list) and obj["steps"]:
                resources = obj.get("resources") if isinstance(obj.get("resources"), dict) else {}
                return obj["steps"], resources
        except ValueError:
            pass
        # object was malformed/truncated: salvage the steps array on its own
        m = re.search(r'"steps"\s*:\s*', clean)
        if m:
            return parse_plan_json(clean[m.end():]), {}
    steps = parse_plan_json(clean)
    if not steps:
        raise ValueError("Parsed plan is empty")
    return steps, {}
