"""Tool registry, shared context and permission handling."""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional



def obj_schema(props: dict, required=()) -> dict:
    return {"type": "object", "properties": props, "required": list(required)}


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    func: Callable
    group: str = "core"
    ask: Any = False                    # bool or callable(args) -> bool
    read_only: bool = False             # allowed in plan mode and in sub-agents
    preview: Optional[Callable] = None  # (ctx, args) -> str shown before asking
    danger: Optional[Callable] = None   # (args) -> bool, forces confirmation even in yolo
    summary_key: Optional[str] = None   # arg shown next to the tool name
    filter_args: bool = True            # drop unknown args (off for MCP tools)

    def spec(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.schema}}

    def needs_ask(self, args: dict) -> bool:
        return bool(self.ask(args)) if callable(self.ask) else bool(self.ask)

    def summarize(self, args) -> str:
        if not isinstance(args, dict):
            return str(args)[:100]
        key = self.summary_key
        if not key:
            for k, v in args.items():
                if isinstance(v, str):
                    key = k
                    break
        v = str(args.get(key, "")) if key else ""
        return v.replace("\n", " ")[:120]


class Registry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}
        self.disabled_groups: set[str] = set()

    def add(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def is_active(self, t: Tool, plan_mode: bool = False) -> bool:
        if t.group in self.disabled_groups:
            return False
        return t.read_only if plan_mode else True

    def active(self, plan_mode: bool = False):
        return [t for t in self.tools.values() if self.is_active(t, plan_mode)]

    def specs(self, plan_mode: bool = False) -> list:
        return [t.spec() for t in self.active(plan_mode)]

    def groups(self) -> dict:
        out: dict[str, list] = {}
        for t in self.tools.values():
            out.setdefault(t.group, []).append(t.name)
        return out

    def tool(self, name, description, props=None, required=(), **kw):
        """Decorator: @reg.tool("name", "desc", {...}, ["req"], group="x", ask=True)."""
        def deco(fn):
            self.add(Tool(name, description, obj_schema(props or {}, required), fn, **kw))
            return fn
        return deco


class Context:
    """Everything tools need: working dir, config, API client, background processes."""

    def __init__(self, cwd, config, client):
        self.cwd = Path(cwd).resolve()
        self.config = config
        self.client = client
        self.todos: list = []
        self.procs: dict = {}
        self.agent = None
        self.interactive = sys.stdin.isatty()
        self.turn_changes: list = []     # [(path, previous_bytes_or_None)] for the current turn
        self.change_history: list = []   # list of finished turns' change lists (for /undo)

    def path(self, p) -> Path:
        pp = Path(str(p)).expanduser()
        if not pp.is_absolute():
            pp = self.cwd / pp
        return pp.resolve()


# Commands that always require confirmation, even with --yolo.
DANGEROUS_SHELL = [re.compile(p, re.I) for p in (
    r"\brm\s+(-[a-z]*\s+)*-[a-z]*r[a-z]*\s+(-[a-z]+\s+)*(/|~|\$HOME|\*)(\s|$|\*|/\*)",
    r"\bmkfs(\.\w+)?\b", r"\bdd\s+[^|;]*\bof=/dev/", r":\(\)\s*\{", r">\s*/dev/(sd|nvme|hd)",
    r"\bchmod\s+-R\s+0?777\s+/(\s|$)", r"\bgit\s+push\s+[^|;]*(--force|-f\b)", r"\bgit\s+reset\s+--hard",
    r"\bgit\s+clean\s+-[a-z]*f", r"\b(curl|wget|iwr|irm|Invoke-WebRequest|Invoke-RestMethod)\b[^|;]*\|\s*(sudo\s+)?(ba|z)?sh\b",
    r"\b(iwr|irm|Invoke-WebRequest|Invoke-RestMethod)\b[^|;]*\|\s*(iex|Invoke-Expression)\b",
    r"\b(rmdir|rd)\s+(/s|/q)[^|;]*\s+[a-z]:\\\s*$", r"\bdel\s+(/[a-z]\s+)*[a-z]:\\\s*(\*|$)",
    r"\bformat\s+[a-z]:", r"\bRemove-Item\b[^|;]*-Recurse[^|;]*\s+(\\|[A-Za-z]:\\?|~)(\s|$)",
    r"\bshutdown\b", r"\bdiskpart\b", r"\breg\s+delete\b", r"\bsudo\s+rm\b",
)]


def shell_is_dangerous(command: str) -> bool:
    return any(rx.search(command) for rx in DANGEROUS_SHELL)


EDIT_TOOLS = {"write_file", "edit_file", "multi_edit"}
SHELL_TOOLS = {"shell", "start_process"}
_MULTI_CMD = {"git", "npm", "pnpm", "yarn", "pip", "pip3", "python", "python3", "py", "cargo", "go", "docker",
              "dotnet", "pytest", "uv", "npx", "make", "pnpx", "poetry", "gh"}
_CHAIN = re.compile(r"&&|\|\||[;|<>`\n]|\$\(")


class Permissions:
    """Permission state: modes (default / accept edits / yolo) plus per-session 'don't ask again' memory."""

    def __init__(self, yolo: bool = False):
        self.yolo = yolo
        self.accept_edits = False
        self.always: set[str] = set()
        self.prefixes: set[str] = set()

    @staticmethod
    def command_prefix(cmd: str) -> str:
        toks = cmd.strip().split()
        if not toks:
            return ""
        if toks[0].lower() in _MULTI_CMD and len(toks) > 1 and not toks[1].startswith("-"):
            return toks[0] + " " + toks[1]
        return toks[0]

    def prefix_allowed(self, cmd: str) -> bool:
        c = cmd.strip()
        if _CHAIN.search(c):
            return False
        return any(c == p or c.startswith(p + " ") for p in self.prefixes)

    def mode(self) -> str:
        return "yolo" if self.yolo else "accept" if self.accept_edits else "default"

    def check(self, tool, args: dict, title: str, body: list, question: str, danger: bool = False) -> bool:
        from .ui import ui
        if not danger:
            if self.yolo or tool.name in self.always:
                return True
            if self.accept_edits and tool.name in EDIT_TOOLS:
                return True
            if tool.name in SHELL_TOOLS and self.prefix_allowed(str(args.get("command", ""))):
                return True
        deny = "No, and tell NimCode what to do differently (esc)"
        if tool.name in EDIT_TOOLS:
            second = "Yes, allow all edits during this session (shift+tab)"
        elif tool.name in SHELL_TOOLS:
            second = "Yes, and don't ask again for `%s` commands this session" % self.command_prefix(str(args.get("command", "")))
        else:
            second = "Yes, and don't ask again for %s this session" % tool.name
        options = ["Yes", deny] if danger else ["Yes", second, deny]
        if not ui.interactive:
            ui.warn("  (%s needs approval but this session is non-interactive: denied; use --yolo)" % tool.name)
            return False
        idx = ui.confirm(title, body, question, options,
                         "This command looks destructive. Confirmation is required even in auto-approve mode." if danger else "")
        if danger:
            return idx == 0
        if idx == 1 and len(options) == 3:
            if tool.name in EDIT_TOOLS:
                self.accept_edits = True
            elif tool.name in SHELL_TOOLS:
                self.prefixes.add(self.command_prefix(str(args.get("command", ""))))
            else:
                self.always.add(tool.name)
        return idx in (0, 1) and idx != len(options) - 1 and idx >= 0
