"""User hooks: shell commands run on PreToolUse / PostToolUse / Stop (configured under "hooks" in ~/.nimcode/config.json).

    "hooks": {
      "PreToolUse":  [{"matcher": "shell",                "command": "python check_cmd.py"}],
      "PostToolUse": [{"matcher": "edit_file|write_file", "command": "ruff format ."}],
      "Stop":        [{"command": "pytest -q"}]
    }

The hook gets NIMCODE_EVENT, NIMCODE_TOOL, NIMCODE_TOOL_INPUT (JSON) and NIMCODE_TOOL_RESULT in its environment.
Exit code 2 = feedback: PreToolUse blocks the tool call, PostToolUse appends the output to the result, Stop sends the
output back to the model so it keeps working. Other exit codes are ignored."""
from __future__ import annotations

import json
import os
import re
import subprocess

from .gitfinder import tool_env
from .tools.shell import shell_argv
from .util import decode_bytes


def run_hooks(config: dict, event: str, cwd, tool: str = "", args=None, result: str = "") -> list:
    """Returns [(exit_code, output)] for matching hooks."""
    out = []
    for h in (config.get("hooks") or {}).get(event, []) or []:
        if not isinstance(h, dict) or not h.get("command"):
            continue
        m = h.get("matcher", "")
        if m and m != "*" and tool and not re.search(m, tool):
            continue
        env = dict(tool_env(), NIMCODE_EVENT=event, NIMCODE_TOOL=tool, NIMCODE_TOOL_INPUT=json.dumps(args or {})[:8000],
                   NIMCODE_TOOL_RESULT=(result or "")[:8000])
        try:
            p = subprocess.run(shell_argv(h["command"]), cwd=str(cwd), env=env, capture_output=True, timeout=int(h.get("timeout", 60)),
                               stdin=subprocess.DEVNULL)
            out.append((p.returncode, (decode_bytes(p.stdout) + decode_bytes(p.stderr)).strip()))
        except (subprocess.TimeoutExpired, OSError) as e:
            out.append((None, "hook failed: %s" % e))
    return out
