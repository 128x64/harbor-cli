"""User extensions: custom Python tools and custom slash commands (markdown prompt templates)."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

from .core import Tool, Registry, obj_schema


def tool_dirs(cwd: Path, include_project: bool) -> list:
    dirs = [Path.home() / ".nimcode" / "tools"]
    if include_project:
        dirs.append(cwd / ".nimcode" / "tools")
    return [d for d in dirs if d.is_dir()]


def command_dirs(cwd: Path, include_project: bool) -> list:
    dirs = [Path.home() / ".nimcode" / "commands"]
    if include_project:
        dirs.append(cwd / ".nimcode" / "commands")
    return [d for d in dirs if d.is_dir()]


def project_extensions(cwd: Path) -> list:
    """Files in the project that would execute code or spawn processes when loaded."""
    found = []
    for rel in (".nimcode/tools", ".mcp.json", ".vscode/mcp.json", ".cursor/mcp.json", ".nimcode/mcp.json"):
        p = cwd / rel
        if p.is_dir() and any(p.glob("*.py")):
            found.append(rel + "/*.py")
        elif p.is_file():
            found.append(rel)
    return found


def _wrap(fn):
    takes_ctx = "ctx" in inspect.signature(fn).parameters

    def w(ctx, **kw):
        res = fn(ctx, **kw) if takes_ctx else fn(**kw)
        return res if isinstance(res, str) else str(res)
    return w


def load_tools(reg: Registry, dirs: list, report=print) -> int:
    """Each *.py defines either register(reg) or TOOLS = [{name, description, parameters, required, func, ask, read_only}]."""
    count = 0
    for d in dirs:
        for f in sorted(d.glob("*.py")):
            try:
                spec = importlib.util.spec_from_file_location("nimcode_plugin_" + f.stem, f)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                before = len(reg.tools)
                if hasattr(mod, "register"):
                    mod.register(reg)
                for t in getattr(mod, "TOOLS", []) or []:
                    reg.add(Tool(name=t["name"], description=t.get("description", ""),
                                 schema=obj_schema(t.get("parameters", {}), t.get("required", [])),
                                 func=_wrap(t["func"]), group="plugins", ask=t.get("ask", True),
                                 read_only=t.get("read_only", False)))
                count += len(reg.tools) - before
            except Exception as e:
                report("plugin %s failed to load: %s: %s" % (f, type(e).__name__, e))
    return count


def load_commands(dirs: list) -> dict:
    cmds = {}
    for d in dirs:
        for f in sorted(d.glob("*.md")):
            try:
                cmds[f.stem.lower()] = f.read_text(encoding="utf-8")
            except OSError:
                pass
    return cmds
