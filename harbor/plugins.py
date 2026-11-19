"""User extensions: custom Python tools and custom slash commands (markdown prompt templates)."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

from .core import Tool, Registry, obj_schema


PROJECT_DIRS = (".harbor", ".nimcode")      # ".nimcode" = the folder name from before the rename; still read


def _dirs(cwd: Path, include_project: bool, sub: str) -> list:
    dirs = [Path.home() / ".harbor" / sub]
    if include_project:
        dirs += [cwd / d / sub for d in PROJECT_DIRS]
    return [d for d in dirs if d.is_dir()]


def tool_dirs(cwd: Path, include_project: bool) -> list:
    return _dirs(cwd, include_project, "tools")


def command_dirs(cwd: Path, include_project: bool) -> list:
    return _dirs(cwd, include_project, "commands")


def project_extensions(cwd: Path) -> list:
    """Files in the project that would execute code or spawn processes when loaded."""
    found = []
    for rel in (".harbor/tools", ".mcp.json", ".vscode/mcp.json", ".cursor/mcp.json", ".harbor/mcp.json", ".harbor/agents", ".harbor/skills",
                ".nimcode/tools", ".nimcode/mcp.json", ".nimcode/agents", ".nimcode/skills"):
        p = cwd / rel
        if p.is_dir() and rel.endswith(("agents", "skills")) and any(p.iterdir()):
            found.append(rel + "/ (prompts that steer the model)")
        elif p.is_dir() and any(p.glob("*.py")):
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
                spec = importlib.util.spec_from_file_location("harbor_plugin_" + f.stem, f)
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


def parse_frontmatter(text: str):
    """'---\nkey: value\n---\nbody' -> ({key: value}, body). Values may be 'a, b' lists for `tools`."""
    text = text.lstrip("\ufeff")
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    meta = {}
    for line in text[3:end].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip().strip("\"'")
    return meta, text[end + 4:].lstrip("\n")


def agent_dirs(cwd: Path, include_project: bool) -> list:
    return _dirs(cwd, include_project, "agents")


def skill_dirs(cwd: Path, include_project: bool) -> list:
    return _dirs(cwd, include_project, "skills")


def load_agents(dirs: list) -> dict:
    """Custom sub-agents: <dir>/<name>.md with front matter (name, description, tools, model) and the prompt as body."""
    out = {}
    for d in dirs:
        for f in sorted(d.glob("*.md")):
            try:
                meta, body = parse_frontmatter(f.read_text(encoding="utf-8"))
            except OSError:
                continue
            name = (meta.get("name") or f.stem).strip().lower().replace(" ", "-")
            tools = [t.strip() for t in meta.get("tools", "").split(",") if t.strip()] or None
            out[name] = {"description": meta.get("description", "(no description)"), "tools": tools,
                         "model": meta.get("model", ""), "prompt": body.strip(), "path": str(f)}
    return out


def load_skills(dirs: list) -> dict:
    """Skills: <dir>/<name>/SKILL.md with front matter (name, description). Loaded on demand by the `skill` tool."""
    out = {}
    for d in dirs:
        for f in sorted(d.glob("*/SKILL.md")):
            try:
                meta, body = parse_frontmatter(f.read_text(encoding="utf-8"))
            except OSError:
                continue
            name = (meta.get("name") or f.parent.name).strip().lower().replace(" ", "-")
            out[name] = {"description": meta.get("description", "(no description)"), "dir": str(f.parent), "path": str(f), "body": body}
    return out
