"""Command-line entry point: argument parsing, config, REPL, slash commands, sessions."""
from __future__ import annotations

import argparse
import atexit
import getpass
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from . import REPO, REPO_URL, VERSION, gitfinder, mcp as mcplib, plugins, updater
from .agent import Agent
from .backends import find_llama_server, find_local_model, make_local_client
from .ollama import OllamaClient, normalize_host
from . import warmup
from .models import apply_profile, profile_for
from .providers import PROVIDERS, find_key, provider_info
from .client import DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_VISION_MODEL, NIMClient, NIMError
from .core import Context, Permissions, Registry
from .textproto import AutoToolClient, TextToolClient
from .tools import register_all
from .tools.shell import shell_name, run_command, shell_argv
from .ui import ui, G, gray, accent
from . import util
from .util import getenv, have

CONFIG_DIR = Path.home() / ".harbor"
CONFIG_FILE = CONFIG_DIR / "config.json"
HISTORY_FILE = CONFIG_DIR / "history.json"
SESSION_DIR = CONFIG_DIR / "sessions"
KEY_ENV_VARS = ("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY", "NIM_API_KEY")

BUILTIN_COMMANDS = [
    ("/help", "Show help and shortcuts"), ("/clear", "Clear conversation history"),
    ("/compact", "Summarize history to free up context"), ("/model", "Show or switch the model"),
    ("/models", "List models available to your key"), ("/vision-model", "Show or set the vision model"),
    ("/status", "Show version, model, tools and settings"), ("/doctor", "Check optional dependencies"),
    ("/tools", "List / enable / disable tool groups"), ("/mcp", "Show MCP servers and tools"),
    ("/mode", "Show or set the mode: manual, accept, plan, auto, bypass"), ("/ui", "Switch terminal look: classic | modern"),
    ("/plan", "Toggle plan mode (read-only)"), ("/yolo", "Toggle bypass (auto-approve everything)"),
    ("/reasoning", "Toggle display of model reasoning"), ("/todos", "Show the task list"),
    ("/cost", "Show token usage"), ("/init", "Create a HARBOR.md for this project"),
    ("/memory", "Show long-term memory"), ("/auto", "Run a goal autonomously until done: /auto <goal>"),
    ("/undo", "Revert the file changes of the last turn"), ("/verify", "Set/clear the verification command"),
    ("/agents", "List custom sub-agents"), ("/skills", "List skills"), ("/permissions", "Show or add allow/ask/deny rules"),
    ("/context", "Show context-window usage"), ("/resume", "Resume an earlier session"), ("/export", "Save this conversation as markdown"),
    ("/review", "Review the current changes"), ("/commit", "Commit the current changes"), ("/test", "Run the tests and fix failures"),
    ("/fix", "Fix a problem: /fix <description>"), ("/explain", "Explain code: /explain <what>"), ("/security-review", "Security review"),
    ("/refactor", "Refactor: /refactor <what>"), ("/docs", "Write documentation"), ("/pr", "Draft a pull request description"), ("/update", "Check for a newer Harbor CLI and install it"), ("/scan", "Scan a repo: folder, GitHub URL or owner/repo"),
    ("/exit", "Exit Harbor CLI"),
]


# ---------------------------------------------------------------- config / sessions
def load_config() -> dict:
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    try:
        os.chmod(CONFIG_FILE, 0o600)
    except OSError:
        pass


def resolve_api_key(cfg: dict, interactive: bool):
    for var in KEY_ENV_VARS:
        if os.environ.get(var):
            return os.environ[var].strip()
    if cfg.get("api_key"):
        return cfg["api_key"].strip()
    if not interactive:
        return None
    print("No NVIDIA API key found. Get a free one at https://build.nvidia.com (profile -> API Keys),\nor start Ollama (ollama.com), or put a model.gguf next to main.py.")
    try:
        key = getpass.getpass("Paste your nvapi-... key (input hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not key:
        return None
    try:
        if input("Save it to %s? [y/N] " % CONFIG_FILE).strip().lower() in ("y", "yes"):
            cfg["api_key"] = key
            save_config(cfg)
            print("Saved.")
    except (EOFError, KeyboardInterrupt):
        pass
    return key


def load_history() -> list:
    try:
        h = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return [x for x in h if isinstance(x, str)][-500:]
    except (OSError, ValueError):
        return []


def save_history(h: list) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(json.dumps(h[-500:]), encoding="utf-8")
    except OSError:
        pass


def session_path(cwd: Path, stamp: str) -> Path:
    import hashlib
    return SESSION_DIR / ("%s-%s.json" % (hashlib.md5(str(cwd).encode()).hexdigest()[:8], stamp))


def save_session(agent: Agent, path: Path) -> None:
    try:
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"cwd": str(agent.ctx.cwd), "model": agent.model, "messages": agent.messages[1:]}), encoding="utf-8")
    except OSError:
        pass


def latest_session(cwd: Path):
    import hashlib
    pref = hashlib.md5(str(cwd).encode()).hexdigest()[:8]
    files = sorted(SESSION_DIR.glob(pref + "-*.json")) if SESSION_DIR.is_dir() else []
    return files[-1] if files else None


# ---------------------------------------------------------------- slash commands
def help_text(custom: dict) -> list:
    lines = [accent(" Harbor CLI %s" % VERSION), "", " Slash commands:"]
    lines += ["   %-14s %s" % c for c in BUILTIN_COMMANDS]
    if custom:
        lines += ["", " Custom commands: " + ", ".join("/" + k for k in sorted(custom))]
    lines += ["", " Shortcuts:", "   ! <cmd>        run a shell command directly", "   esc            interrupt the current response",
              "   shift+tab / ctrl+t   cycle mode: manual > accept edits > plan > autonomous (or /mode <name>)", "   alt+enter or \\+enter   newline",
              "   @path          mention a file (Tab completes)      up / down  history", "   ctrl+c         clear input; twice to exit"]
    return lines


MODE_ORDER = ["manual", "accept", "plan", "auto"]
MODE_INFO = {   # key -> (label, description)
    "manual": ("MANUAL MODE", "you approve every edit and command"),
    "accept": ("ACCEPT EDITS", "edits auto-approved, commands still ask"),
    "plan": ("PLAN MODE", "read-only: investigates and proposes a plan"),
    "auto": ("AUTONOMOUS MODE", "works until done and verified; edits auto-approved"),
    "bypass": ("BYPASS PERMISSIONS", "everything auto-approved (destructive still asks)"),
    "autobypass": ("AUTONOMOUS + BYPASS", "fully hands-off (destructive commands still ask)"),
}


def current_mode(agent: Agent) -> str:
    return agent.mode_key()


def mode_status(agent: Agent):
    key = agent.mode_key()
    label, desc = MODE_INFO[key]
    return key, label, desc


def set_mode(agent: Agent, key: str) -> None:
    """Set the permission/autonomy mode: manual | accept | plan | auto | bypass | autobypass."""
    agent.set_mode(key)


def cycle_mode(agent: Agent, allow_bypass: bool = False) -> None:
    order = MODE_ORDER + (["bypass"] if allow_bypass else [])
    cur = agent.mode_key()
    nxt = order[(order.index(cur) + 1) % len(order)] if cur in order else order[0]
    set_mode(agent, nxt)


def _ollama_line() -> str:
    try:
        oc = OllamaClient("")
        if not oc.is_up(1.0):
            return "not running at %s (start it, or ignore if you use NIM)" % oc.host
        names = oc.list_models()
        return "running at %s - %d model(s)%s" % (oc.host, len(names), ": " + ", ".join(names[:5]) if names else " (run: ollama pull qwen2.5-coder:7b)")
    except Exception as e:  # noqa: BLE001
        return "error: %s" % e


def doctor_lines(agent) -> list:
    deps = [("pygments", "pygments", "syntax highlighting"), ("pypdf", "pypdf", "read PDFs"), ("docx", "python-docx", "read DOCX"),
            ("openpyxl", "openpyxl", "read XLSX"), ("pptx", "python-pptx", "read PPTX"), ("bs4", "beautifulsoup4", "HTML + Bing parsing"),
            ("markdownify", "markdownify", "HTML to markdown"), ("psutil", "psutil", "system info/processes"),
            ("pyperclip", "pyperclip", "clipboard tools"), ("PIL", "pillow", "screenshots + image resizing"),
            ("uv", "uv", "uvx-style MCP servers without Node"), ("dulwich", "dulwich", "pure-Python git fallback"),
            ("llama_cpp", "llama-cpp-python", "run model.gguf in-process (alternative to llama-server)"),
            ("playwright", "playwright", "browser_* tools (then: playwright install chromium)"),
            ("rapidocr_onnxruntime", "rapidocr-onnxruntime", "local OCR")]
    root = Path(__file__).resolve().parent.parent
    gguf = find_local_model([root])
    out = ["Python %s on %s" % (sys.version.split()[0], sys.platform), "Shell tool: %s" % shell_name(),
           "git: %s" % (gitfinder.find_git() or "NOT FOUND (install Git, or GitHub Desktop - its bundled git is used automatically)"),
           "Ollama: %s" % _ollama_line(),
           "Local model: %s" % (gguf or "none (no model.gguf next to main.py -> NVIDIA NIM is used)"),
           "llama-server: %s" % (find_llama_server([root]) or "not found (needed for model.gguf unless llama-cpp-python is installed)"), ""]
    for mod, pipname, what in deps:
        ok = have(mod)
        out.append("  %s %-22s %s" % ("ok     " if ok else "missing", pipname, what))
    out += ["", "Install everything: pip install -r requirements.txt   (see requirements-local.txt / requirements-extra.txt)"]
    return out


def mask(key: str) -> str:
    return key[:9] + "..." + key[-4:] if len(key) > 16 else "(set)"


def status_lines(agent: Agent, args, cfg, mcp_clients) -> list:
    reg = agent.registry
    return ["Version:       Harbor CLI %s" % VERSION, "Backend:       %s" % state_backend(agent),
            "Model:         %s" % agent.model, "Vision model:  %s" % (cfg.get("vision_model") or DEFAULT_VISION_MODEL),
            "Endpoint:      %s" % agent.client.base_url, "API key:       %s" % (mask(agent.client.api_key) if agent.client.api_key != "local" else "(local model)"),
            "Working dir:   %s" % agent.ctx.cwd, "Mode:          %s%s" % (current_mode(agent), " + autonomous" if agent.auto else ""),
            "Verify cmd:    %s" % (agent.verify_cmd or "(none)"), "Context window: %d tokens" % agent.context_window,
            "Updates:       %s from %s (%s)" % (cfg.get("auto_update", "auto"), cfg.get("update_repo") or REPO, updater.install_kind()),
            "Shell:         %s" % shell_name(), "git:           %s" % (gitfinder.find_git() or "not found (pure-Python fallback if dulwich installed)"),
            "Tools:         %d active of %d" % (len(agent.active_tools()), len(reg.tools)),
            "MCP servers:   %s" % (", ".join("%s (%d)" % (c.name, len(c.tools)) for c in mcp_clients) or "none")]


def state_backend(agent) -> str:
    return getattr(agent.client, "backend_desc", None) or ("NVIDIA NIM" if "nvidia" in str(agent.client.base_url) else str(agent.client.base_url))


BUILTIN_PROMPTS = {
    "review": ("Review the current changes: run `git status` and `git diff` (staged and unstaged); if there are none, review the latest commit "
               "(`git show`). Read surrounding code where needed. Report real problems first (bugs, security, edge cases, missing tests) "
               "ordered by severity with path:line references, then brief style notes. Do not modify any files. $ARGUMENTS"),
    "commit": ("Create a git commit for the current changes: run `git status`, `git diff` and `git log -5 --oneline` (match the message style); "
               "stage only the relevant files (never secrets or build artifacts); write a concise message explaining WHY; commit. Do not push. $ARGUMENTS"),
    "test": ("Find how this project runs its tests (package files, Makefile, CI config), run them and report. If anything fails, diagnose the "
             "root cause, fix it and re-run until green. $ARGUMENTS"),
    "fix": ("Fix this problem: $ARGUMENTS. Reproduce it first, find the root cause, make the smallest correct change and verify with the "
            "relevant tests or commands."),
    "explain": ("Explain $ARGUMENTS for a developer new to this code: purpose, how it works, key files with path:line references. "
                "Read the code first; do not guess."),
    "security-review": ("Security-review $ARGUMENTS (default: the whole project): injection, unsafe deserialization, path traversal, secrets in "
                        "code, weak auth, risky dependencies. Report findings with severity, path:line and a concrete fix. Do not modify files."),
    "refactor": "Refactor $ARGUMENTS to be simpler and clearer without changing behaviour. Run the tests before and after.",
    "docs": ("Write or update documentation for $ARGUMENTS (README sections or docstrings) in the project's existing style. "
             "Verify that any commands you document actually work."),
    "scan": ("Scan $ARGUMENTS with the scan_repo tool (use '.' if no target was given; it accepts a folder, a GitHub URL or owner/repo), then "
             "summarize what the project is, how to run and test it, and the most important security or quality findings with suggested fixes. "
             "Do not modify any files."),
    "pr": ("Prepare a pull request description for the current branch: use `git log` and `git diff` against the main branch to summarize what "
           "changed and why, how it was tested and the risks. Output markdown only; do not push anything. $ARGUMENTS"),
}

INIT_PROMPT = ("Explore this project (structure, build/test/run commands, conventions) and create a concise HARBOR.md in the "
               "project root with guidance for future coding sessions. Keep it under 80 lines.")


def context_lines(agent: Agent) -> list:
    est = lambda s: int(len(s) / 3.5)
    sys_t = est(agent.messages[0]["content"])
    tools = agent.active_tools()
    tool_t = est(json.dumps([t.spec() for t in tools]))
    msgs = agent.messages[1:]
    conv_t = est(json.dumps(msgs, ensure_ascii=False))
    total, win = sys_t + tool_t + conv_t, agent.context_window
    pct = min(100, int(100 * total / max(1, win)))
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    fmt = lambda n: format(n, ",")
    return ["Context usage  ~%s / %s tokens (%d%%)" % (fmt(total), fmt(win), pct), "  " + bar,
            "  system prompt   %8s" % fmt(sys_t), "  tool schemas    %8s  (%d tools)" % (fmt(tool_t), len(tools)),
            "  conversation    %8s  (%d messages)" % (fmt(conv_t), len(msgs)), "  free            %8s" % fmt(max(0, win - total)),
            gray("  Tip: /compact summarizes history; /tools off <group> trims the tool list.")]


def list_sessions(cwd: Path) -> list:
    import hashlib
    pref = hashlib.md5(str(cwd).encode()).hexdigest()[:8]
    out = []
    for f in sorted(SESSION_DIR.glob(pref + "-*.json"), reverse=True)[:12] if SESSION_DIR.is_dir() else []:
        try:
            msgs = json.loads(f.read_text(encoding="utf-8")).get("messages", [])
        except (OSError, ValueError):
            continue
        first = next((m["content"] for m in msgs if m.get("role") == "user" and isinstance(m.get("content"), str)
                      and not m["content"].startswith("[")), "")
        out.append((f, len(msgs), " ".join(first.split())[:50]))
    return out


def export_markdown(agent: Agent) -> str:
    out = ["# Harbor CLI session", "", "- folder: %s" % agent.ctx.cwd, "- model: %s" % agent.model, "- exported: %s" % time.strftime("%Y-%m-%d %H:%M:%S"), ""]
    for m in agent.messages[1:]:
        role, content = m["role"], m.get("content") or ""
        if role == "user":
            out += ["## You", "", content if isinstance(content, str) else json.dumps(content), ""]
        elif role == "assistant":
            if content:
                out += ["## Harbor CLI", "", content, ""]
            for tc in m.get("tool_calls") or []:
                out += ["```", "tool: %s %s" % (tc["function"]["name"], tc["function"]["arguments"][:600]), "```", ""]
        elif role == "tool":
            out += ["<details><summary>result</summary>", "", "```", str(content)[:1500], "```", "", "</details>", ""]
    return "\n".join(out)


def handle_command(agent: Agent, line: str, state: dict) -> str:
    """Return "quit", "prompt:<text>" to send a prompt to the model, or "" when handled."""
    cmd, _, arg = line[1:].partition(" ")
    cmd, arg = cmd.lower(), arg.strip()
    cfg = state["cfg"]
    if cmd in ("exit", "quit", "q"):
        return "quit"
    if cmd == "help":
        for ln in help_text(state["custom"]):
            ui.line(ln)
    elif cmd == "clear":
        agent.reset()
        ui.info("Conversation cleared.")
    elif cmd == "compact":
        try:
            agent.compact()
        except NIMError as e:
            ui.error("Error: %s" % e.message)
    elif cmd == "model":
        if arg:
            agent.model = arg
            ui.info("Model set to %s" % arg)
        else:
            ui.line(agent.model)
    elif cmd == "vision-model":
        if arg:
            cfg["vision_model"] = arg
            ui.info("Vision model set to %s" % arg)
        else:
            ui.line(cfg.get("vision_model") or DEFAULT_VISION_MODEL)
    elif cmd == "models":
        try:
            ids = [m for m in agent.client.list_models() if arg.lower() in m.lower()]
            ui.line("\n".join(ids[:300]) or "No models matched.")
            ui.info("%d models" % len(ids))
        except NIMError as e:
            ui.error("Error: %s" % e.message)
    elif cmd == "status":
        for ln in status_lines(agent, state["args"], cfg, state["mcp"]):
            ui.line(ln)
    elif cmd == "doctor":
        for ln in doctor_lines(agent):
            ui.line(ln)
    elif cmd == "tools":
        reg = agent.registry
        parts = arg.split()
        if len(parts) == 2 and parts[0] in ("on", "off"):
            if parts[1] not in reg.groups():
                ui.error("Unknown group '%s'." % parts[1])
            elif parts[0] == "off":
                reg.disabled_groups.add(parts[1])
                agent.refresh_system_prompt()
            else:
                reg.disabled_groups.discard(parts[1])
                agent.refresh_system_prompt()
        for g, names in sorted(reg.groups().items()):
            on = g not in reg.disabled_groups
            ui.line("  %s %-14s %s" % ("on " if on else "off", g, gray(", ".join(names) if g.startswith("mcp:") is False else "%d tools" % len(names))))
        ui.info("Use: /tools off <group>  or  /tools on <group>")
    elif cmd == "mcp":
        if not state["mcp"]:
            ui.info("No MCP servers connected. Add them to ~/.harbor/mcp.json or .mcp.json (see README).")
        for c in state["mcp"]:
            ui.line("  %s (%d tools)" % (c.name, len(c.tools)))
            for t in c.tools[:30]:
                ui.line(gray("     %s" % t.get("name")))
    elif cmd == "update":
        info, status = updater.check(cfg, force=True)
        if status != "available":
            ui.info({"current": "Harbor CLI %s is up to date." % VERSION, "unreachable": "Could not reach GitHub.",
                     "bad-repo": "update_repo must be a github.com owner/repo."}.get(status, "No update available (%s)." % status))
        elif updater.install_kind() != "folder":
            ui.warn("Harbor CLI %s is available, but this is a %s install: %s" % (
                info["version"], updater.install_kind(), "run `git pull`" if updater.install_kind() == "git" else "update it with pip"))
        else:
            body = ["Harbor CLI %s -> %s" % (VERSION, info["version"]), ""] + (info.get("notes") or "").strip().splitlines()[:14]
            idx = ui.confirm("Update available", body, "Install now? Harbor CLI restarts afterwards.",
                             ["Yes, install and restart", "Not now", "Skip this version"])
            if idx == 0:
                state["pending_update"] = info
                return "update"
            if idx == 2:
                updater.skip_version(info["version"])
    elif cmd == "ui":
        if arg in ("classic", "modern"):
            ui.style = arg
        ui.info("UI style: %s  (/ui classic | /ui modern)" % ui.style)
    elif cmd == "mode":
        key = arg.lower().replace(" ", "")
        alias = {"default": "manual", "ask": "manual", "edits": "accept", "acceptedits": "accept", "autonomous": "auto", "yolo": "bypass"}
        key = alias.get(key, key)
        if key in MODE_INFO:
            set_mode(agent, key)
        elif arg:
            ui.error("Unknown mode '%s'. Modes: manual, accept, plan, auto, bypass." % arg)
        for k in MODE_ORDER + ["bypass"]:
            ui.line("  %s %-20s %s" % ("●" if k == agent.mode_key() else " ", MODE_INFO[k][0], gray(MODE_INFO[k][1])))
        ui.info("Switch with shift+tab / ctrl+t, or /mode <manual|accept|plan|auto|bypass>.")
    elif cmd == "plan":
        set_mode(agent, "manual" if agent.plan_mode else "plan")
        ui.info("Now in %s." % MODE_INFO[agent.mode_key()][0])
    elif cmd == "yolo":
        set_mode(agent, "manual" if agent.perms.yolo else ("autobypass" if agent.auto else "bypass"))
        ui.info("Now in %s." % MODE_INFO[agent.mode_key()][0])
    elif cmd == "reasoning":
        ui.show_reasoning = not ui.show_reasoning
        ui.info("Reasoning display %s." % ("on" if ui.show_reasoning else "off"))
    elif cmd == "todos":
        lines, _ = ui.result_lines("todo_write", {}, "", agent.ctx.todos)
        for ln in (lines if agent.ctx.todos else [gray("No todos.")]):
            ui.line("  " + ln)
    elif cmd in ("cost", "tokens"):
        ui.line("requests: %d | prompt tokens: %d | completion tokens: %d | context now: ~%d / %d"
                % (agent.requests, agent.prompt_tokens, agent.completion_tokens, agent.estimate_tokens(), agent.context_window))
    elif cmd == "init":
        return "prompt:" + INIT_PROMPT
    elif cmd == "auto":
        if not arg:
            set_mode(agent, "manual" if agent.auto else ("autobypass" if agent.perms.yolo else "auto"))
            ui.info("Now in %s." % MODE_INFO[agent.mode_key()][0])
        else:
            state["auto_prev"] = agent.mode_key()
            set_mode(agent, "autobypass" if agent.perms.yolo else "auto")
            return "prompt:" + arg
    elif cmd == "undo":
        restored = agent.undo()
        if restored:
            for pth in restored:
                ui.line("  restored " + ui.rel(pth))
        else:
            ui.info("Nothing to undo (only file edits made through Harbor CLI's tools are tracked).")
    elif cmd == "verify":
        agent.verify_cmd = arg
        ui.info("Verification command: %s" % (arg or "(cleared)"))
    elif cmd == "memory":
        from .tools.agentic import MEMORY_FILE
        try:
            ui.line(MEMORY_FILE.read_text(encoding="utf-8") or "(empty)")
        except OSError:
            ui.info("No memory yet. Ask the agent to remember something.")
        ui.info("File: %s" % MEMORY_FILE)
    elif cmd in state["custom"]:
        return "prompt:" + state["custom"][cmd].replace("$ARGUMENTS", arg)
    elif cmd in BUILTIN_PROMPTS:
        return "prompt:" + BUILTIN_PROMPTS[cmd].replace("$ARGUMENTS", arg).strip()
    elif cmd == "agents":
        if not agent.ctx.agents:
            ui.info("No custom sub-agents. Create .md files in ~/.harbor/agents/ or <project>/.harbor/agents/ (see examples/agents).")
        for n, a_ in agent.ctx.agents.items():
            ui.line("  %-18s %s %s" % (n, a_["description"], gray("[tools: %s]" % (", ".join(a_["tools"]) if a_["tools"] else "read-only"))))
    elif cmd == "skills":
        if not agent.ctx.skills:
            ui.info("No skills. Create <dir>/<name>/SKILL.md in ~/.harbor/skills/ or <project>/.harbor/skills/ (see examples/skills).")
        for n, s_ in agent.ctx.skills.items():
            ui.line("  %-18s %s" % (n, s_["description"]))
    elif cmd == "permissions":
        parts = arg.split(None, 1)
        if len(parts) == 2 and parts[0] in ("allow", "ask", "deny"):
            agent.perms.add_rule(parts[0], parts[1])
            ui.info("Added %s rule for this session: %s" % (parts[0], parts[1]))
        for kind in ("deny", "ask", "allow"):
            for r in agent.perms.rules.get(kind, []):
                ui.line("  %-6s %s" % (kind, r))
        if not any(agent.perms.rules.values()):
            ui.info("No rules. Add to config.json:  \"permissions\": {\"allow\": [\"shell(git status*)\"], \"deny\": [\"read_file(.env*)\"]}")
        ui.info("Usage: /permissions allow|ask|deny tool(pattern)   e.g.  /permissions deny shell(rm *)")
    elif cmd == "context":
        for ln in context_lines(agent):
            ui.line(ln)
    elif cmd == "resume":
        sessions = list_sessions(agent.ctx.cwd)
        if not sessions:
            ui.info("No saved sessions for this folder.")
        else:
            labels = ["%s · %d msgs · %s" % (f.stem.split("-", 1)[1], n, first) for f, n, first in sessions]
            pick = -1
            if arg.isdigit() and 1 <= int(arg) <= len(sessions):
                pick = int(arg) - 1
            elif ui.interactive:
                pick = ui.confirm("Resume a session", ["Saved sessions in this folder (newest first):"], "Which one?", labels[:9])
            else:
                for i, lb in enumerate(labels, 1):
                    ui.line("  %d. %s" % (i, lb))
                ui.info("Use /resume <number>.")
            if pick >= 0:
                try:
                    data = json.loads(sessions[pick][0].read_text(encoding="utf-8"))
                    agent.messages = [agent.messages[0]] + data.get("messages", [])
                    state["session_file"] = sessions[pick][0]
                    ui.info("Resumed %s (%d messages)." % (sessions[pick][0].stem, len(data.get("messages", []))))
                except (OSError, ValueError):
                    ui.error("Could not read that session.")
    elif cmd == "export":
        path = Path(arg).expanduser() if arg else agent.ctx.cwd / ("harbor-export-%s.md" % time.strftime("%Y%m%d-%H%M%S"))
        try:
            path.write_text(export_markdown(agent), encoding="utf-8")
            ui.info("Saved %s" % path)
        except OSError as e:
            ui.error("Could not write %s: %s" % (path, e))
    else:
        ui.error("Unknown command /%s. Try /help." % cmd)
    return ""


# ---------------------------------------------------------------- running turns
def run_safely(agent: Agent, text: str) -> bool:
    try:
        agent.run_turn(text)
        return True
    except KeyboardInterrupt:
        ui.line()
        ui.tool_result(["Interrupted by user"], error=True)
        if agent.messages and agent.messages[-1]["role"] == "user":
            agent.messages.pop()
    except NIMError as e:
        ui.error("NIM API error%s: %s" % (" (%d)" % e.status if e.status else "", e.message))
        if e.status in (401, 403):
            ui.info("  Check your key (starts with nvapi-) at https://build.nvidia.com")
        elif e.status == 404:
            ui.info("  Model '%s' not found for your key. Use /models to list available ones." % agent.model)
        elif e.status in (400, 422) and re.search(r"tool", e.message, re.I):
            ui.info("  This model may not support tool calling. Try /model nvidia/nemotron-3-super-120b-a12b "
                    "or /model qwen/qwen3-coder-480b-a35b-instruct")
        if agent.messages and agent.messages[-1]["role"] == "user":
            agent.messages.pop()
    return False


def make_file_lister(cwd: Path):
    """@file completion: relative paths under cwd matching the typed prefix (cached for a few seconds)."""
    cache = {"t": 0.0, "files": []}
    from .tools.fs import walk_files

    def lister(prefix: str) -> list:
        if time.time() - cache["t"] > 5:
            files = []
            for i, f in enumerate(walk_files(cwd)):
                if i > 4000:
                    break
                files.append(f.relative_to(cwd).as_posix())
            cache.update(t=time.time(), files=files)
        p = prefix.lower().replace("\\", "/")
        hits = [f for f in cache["files"] if p in f.lower()]
        hits.sort(key=lambda f: (not f.lower().startswith(p) and not Path(f).name.lower().startswith(p), len(f), f))
        return hits[:7]
    return lister


def repl(agent: Agent, state: dict, first_prompt: str, session_file: Path) -> None:
    cfg = state["cfg"]
    state["session_file"] = session_file
    key, label, desc = mode_status(agent)
    rows = [("backend", state_backend(agent)), ("model", agent.model), ("folder", str(agent.ctx.cwd)),
            ("mode", "%s  %s" % (label, gray("- shift+tab to switch"))),
            ("tools", "%d active%s" % (len(agent.active_tools()), " · MCP: " + ", ".join("%s (%d)" % (c.name, len(c.tools)) for c in state["mcp"]) if state["mcp"] else ""))]
    if not gitfinder.find_git():
        rows.append(("git", "not found - install Git or GitHub Desktop (see /doctor)"))
    ui.welcome(VERSION, rows, key)
    ui.set_title("Harbor CLI - %s" % agent.ctx.cwd.name)
    history = load_history()
    commands = BUILTIN_COMMANDS + [("/" + k, "Custom command") for k in sorted(state["custom"])]
    file_lister = make_file_lister(agent.ctx.cwd)
    allow_bypass = bool(state["args"].yolo)

    def right_text():
        pct = int(100 * agent.estimate_tokens() / max(1, agent.context_window))
        short = agent.model.split("/")[-1]
        return short + " · %d%% ctx" % pct

    if state["args"].preload and not first_prompt:           # load a local model now, not when the first message is sent
        secs = None
        try:
            with ui.working("Loading model into memory"):
                secs = warmup.preload(agent)
        except KeyboardInterrupt:
            ui.info("Skipped model preload (it will load on the first message).")
        except NIMError as e:
            ui.warn("Model preload failed (%s); it will load on the first message." % e.message[:100])
        if secs is not None:
            ui.info("Model loaded into memory in %.1fs." % secs)
    pending = first_prompt
    while True:
        if pending:
            text, pending = pending, ""
        else:
            text = ui.read_input(lambda: mode_status(agent), lambda: cycle_mode(agent, allow_bypass), history, commands, right_text, file_lister)
            if text is None:
                break
        text = text.strip()
        if not text:
            continue
        if not history or history[-1] != text:
            history.append(text)
        ui.user_echo(text)
        if text.lower() in ("exit", "quit"):
            break
        if text.startswith("/"):
            res = handle_command(agent, text, state)
            if res == "quit":
                break
            if res == "update":
                save_history(history)
                save_session(agent, state["session_file"])
                result = updater.run_installer(state.pop("pending_update"), updater.install_root(), "auto")
                if result.get("status") == "updated":
                    updater.relaunch(VERSION)
                elif result.get("status") == "failed":
                    ui.error("Update failed: %s" % result.get("error"))
                continue
            if not res.startswith("prompt:"):
                continue
            text = res[7:]
        elif text.startswith("!"):
            ui.tool_header("shell", text[1:].strip())
            try:
                out = run_command(shell_argv(text[1:].strip()), agent.ctx.cwd, 120)
            except KeyboardInterrupt:
                out = "Error: interrupted"
            lines, err = ui.result_lines("shell", {}, out)
            ui.tool_result(lines, err)
            continue
        t_start = time.time()
        ok = run_safely(agent, text)
        if state.get("auto_prev"):              # /auto <goal> is one-shot: go back to the previous mode
            set_mode(agent, state.pop("auto_prev"))
        if time.time() - t_start > 20:
            ui.bell()
        if ok:
            save_session(agent, state["session_file"])
    save_history(history)


# ---------------------------------------------------------------- startup
def ask_mcp_input(iid: str, d: dict):
    """Prompt for a VS Code-style ${input:id} value (password inputs are hidden and never saved)."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    label = d.get("description") or iid
    try:
        if d.get("password"):
            return getpass.getpass("MCP input '%s' (hidden): " % label)
        return input("MCP input '%s'%s: " % (label, " [%s]" % d["default"] if d.get("default") is not None else ""))
    except (EOFError, KeyboardInterrupt):
        return None


def build_registry(cwd: Path, cfg: dict, args) -> tuple:
    reg = Registry()
    register_all(reg)
    include_project = False
    ext = plugins.project_extensions(cwd)
    if ext:
        trusted = cfg.setdefault("trusted_projects", [])
        if args.trust_project or str(cwd) in trusted:
            include_project = True
        elif ui.interactive and not args.print_mode:
            idx = ui.confirm("Trust this project?", ["This project defines extensions that run code on your machine:", ""] + ["  " + e for e in ext],
                             "Load them?", ["Yes, trust this project", "No, skip them"])
            if idx == 0:
                include_project = True
                trusted.append(str(cwd))
                save_config(cfg)
        else:
            ui.warn("Skipping project extensions (%s); use --trust-project to load them." % ", ".join(ext))
    plugins.load_tools(reg, plugins.tool_dirs(cwd, include_project), report=ui.warn)
    clients = []
    if not args.no_mcp:
        sources = mcplib.discover_sources(cwd, include_project, not args.no_vscode_mcp, [args.mcp_config] if args.mcp_config else [])
        specs = mcplib.resolve_specs(sources, cwd, ask_mcp_input, ui.warn) if sources else {}
        if specs:
            clients = mcplib.connect_servers(specs, cwd, report=ui.info)
            mcplib.register_mcp_tools(reg, clients)
            atexit.register(lambda: [c.close() for c in clients])
    custom = plugins.load_commands(plugins.command_dirs(cwd, include_project))
    agents = plugins.load_agents(plugins.agent_dirs(cwd, include_project))
    skills = plugins.load_skills(plugins.skill_dirs(cwd, include_project))
    if args.tools == "core":
        reg.disabled_groups |= {g for g in reg.groups() if g != "core"}
    for g in cfg.get("disabled_groups", []):
        reg.disabled_groups.add(g)
    return reg, clients, custom, agents, skills


def wrap_tools(client, mode: str, report):
    """Apply --tool-mode: native, text (prompt-based) or auto (native with automatic fallback)."""
    if mode == "native":
        return client
    return TextToolClient(client) if mode == "text" else AutoToolClient(client, report)


def choose_ollama(args, cfg):
    n_ctx = args.n_ctx or int(cfg.get("ollama_num_ctx", 16384))
    cl = OllamaClient(args.ollama_url or cfg.get("ollama_url", ""), n_ctx)
    models = cl.list_models()          # raises a helpful NIMError when the server is not running
    model = args.model or os.environ.get("OLLAMA_MODEL") or cfg.get("ollama_model") or ""
    if not model and not args.list_models:
        if not models:
            raise NIMError(0, "Ollama is running but has no models. Pull one first, e.g.:  ollama pull qwen2.5-coder:7b")
        likes = ("coder", "qwen3", "qwen2.5", "gpt-oss", "llama3.3", "llama3.1", "mistral", "deepseek")
        ranked = sorted(models, key=lambda m: (not any(k in m.lower() for k in likes), m))
        model = ranked[0]
        if len(ranked) > 1 and ui.interactive and not args.print_mode:
            idx = ui.confirm("Choose an Ollama model", ["Installed on %s:" % cl.host], "Which one should Harbor CLI use?", ranked[:9])
            model = ranked[idx] if idx >= 0 else ranked[0]
        ui.info("Ollama model: %s (change with -m or /model)" % model)
    elif model and models and model not in models and model + ":latest" not in models:
        ui.warn("Model '%s' is not installed on this Ollama server (installed: %s). Run: ollama pull %s" % (model, ", ".join(models[:6]) or "none", model))
    client = wrap_tools(cl, args.tool_mode, ui.info)
    client.backend_desc = "Ollama (%s)" % cl.host
    return client, model, client.backend_desc, n_ctx


def choose_provider(args, cfg):
    """Any OpenAI-compatible service via a preset (--provider openrouter, lmstudio, ...)."""
    info = provider_info(args.provider)
    base = args.base_url if args.base_url != DEFAULT_BASE_URL else info["base_url"]
    if not base:
        raise NIMError(0, "--provider custom needs --base-url http://host:port/v1")
    key = find_key(args.provider, cfg, args.api_key_env, args.api_key)
    if not key and info["needs_key"]:
        if sys.stdin.isatty() and not args.print_mode:
            try:
                key = getpass.getpass("Paste your %s API key (hidden): " % info["label"]).strip()
            except (EOFError, KeyboardInterrupt):
                key = ""
        if not key:
            raise NIMError(0, "no API key for %s: set %s or pass --api-key-env/--api-key" % (info["label"], info["key_env"][0] if info["key_env"] else "an env var"))
    cl = NIMClient(key or "none", base)
    client = wrap_tools(cl, args.tool_mode if args.tool_mode != "auto" else "auto", ui.info)
    client.backend_desc = "%s (%s)" % (info["label"], base)
    model = args.model or getenv("MODEL") or cfg.get("model") or info["model"]
    if not model:
        try:
            ids = cl.list_models()
            model = ids[0] if ids else ""
        except NIMError:
            pass
        if not model:
            raise NIMError(0, "pass -m <model id> (could not list models from %s)" % base)
    return client, model, client.backend_desc, None


def choose_backend(args, cfg, main_dir: Path):
    """Return (client, model_name, description, n_ctx_or_None).
    Order: model.gguf next to the main script > explicit --backend > NVIDIA key > running Ollama > ask for a key."""
    pkg_root = Path(__file__).resolve().parent.parent
    dirs = [main_dir, pkg_root, Path.cwd()]
    gguf = find_local_model([main_dir, pkg_root], args.gguf) if args.backend in ("auto", "local") else None
    if args.backend == "local" and not gguf:
        raise NIMError(0, "no model.gguf found next to the script (%s)%s" % (main_dir, " and --gguf path does not exist" if args.gguf else ""))
    if gguf:
        ui.info("Local model found: %s - using it instead of NVIDIA NIM (pass --backend nim to override)." % gguf)
        n_ctx = args.n_ctx or int(cfg.get("n_ctx", 8192))
        client, desc = make_local_client(gguf, args.local_engine, dirs, n_ctx, args.gpu_layers, args.threads, args.tool_mode, ui.info)
        client.backend_desc = "local GGUF via " + desc
        return client, gguf.name, client.backend_desc, n_ctx
    if args.provider and args.provider != "nim" and args.backend in ("auto", "nim"):
        return choose_provider(args, cfg)
    use_ollama = args.backend == "ollama"
    if args.backend == "auto":
        has_key = any(os.environ.get(v) for v in KEY_ENV_VARS) or cfg.get("api_key")
        if not has_key and OllamaClient(args.ollama_url or cfg.get("ollama_url", "")).is_up():
            ui.info("No NVIDIA key found but Ollama is running - using it (see --backend).")
            use_ollama = True
    if use_ollama:
        return choose_ollama(args, cfg)
    key = resolve_api_key(cfg, sys.stdin.isatty() and not args.print_mode)
    if not key:
        raise NIMError(0, "set NVIDIA_API_KEY (free key from https://build.nvidia.com), start Ollama, or put a model.gguf next to main.py")
    if not key.startswith("nvapi-") and args.base_url == DEFAULT_BASE_URL:
        print("Warning: NVIDIA keys normally start with 'nvapi-'.", file=sys.stderr)
    client = NIMClient(key, args.base_url)
    if args.tool_mode == "text":
        client = TextToolClient(client)
    client.backend_desc = "NVIDIA NIM" if args.base_url == DEFAULT_BASE_URL else "OpenAI-compatible (%s)" % args.base_url
    return client, args.model or getenv("MODEL") or cfg.get("model") or DEFAULT_MODEL, client.backend_desc, None


def main(argv=None, main_dir=None) -> int:
    util.init_terminal()
    migrated = util.migrate_legacy_home()
    main_dir = Path(main_dir).resolve() if main_dir else Path(__file__).resolve().parent.parent
    cfg = load_config()
    ap = argparse.ArgumentParser(prog="harbor", description="Agentic coding CLI. Uses a local model.gguf next to the script if present, else NVIDIA NIM.")
    ap.add_argument("prompt", nargs="*", help="initial prompt (the session stays interactive afterwards)")
    ap.add_argument("-p", "--print", dest="print_mode", action="store_true", help="run the prompt non-interactively and exit")
    ap.add_argument("-c", "--continue", dest="cont", action="store_true", help="continue the most recent session in this directory")
    ap.add_argument("-a", "--auto", action="store_true", help="autonomous mode: keep working until the goal is done and verified (calls finish)")
    ap.add_argument("--verify", default=cfg.get("verify", ""), help="command that must pass (exit 0) before a task counts as done, e.g. 'pytest -q'")
    ap.add_argument("--max-minutes", type=float, default=0, help="time budget per request (0 = unlimited)")
    ap.add_argument("-m", "--model", default="", help="model id (NIM) or name (Ollama, e.g. qwen2.5-coder:7b)")
    ap.add_argument("--backend", choices=["auto", "nim", "local", "ollama"], default=cfg.get("backend", "auto"),
                    help="auto: model.gguf next to the script, else NVIDIA NIM (key), else a running Ollama")
    ap.add_argument("--ollama-url", default="", help="Ollama server (default http://localhost:11434 or $OLLAMA_HOST)")
    ap.add_argument("--provider", default=cfg.get("provider", ""), choices=[""] + list(PROVIDERS),
                    help="any OpenAI-compatible service: openai, anthropic, gemini, openrouter, groq, together, deepseek, mistral, xai, "
                         "fireworks, lmstudio, vllm, llamacpp, custom (+ --base-url)")
    ap.add_argument("--api-key-env", default="", help="name of the environment variable holding the provider's API key")
    ap.add_argument("--api-key", default="", help="API key (prefer the environment variable)")
    ap.add_argument("--gguf", default=getenv("GGUF"), help="path to a GGUF model (overrides model.gguf discovery)")
    ap.add_argument("--local-engine", choices=["auto", "server", "python"], default="auto", help="llama-server binary or llama-cpp-python")
    ap.add_argument("--n-ctx", type=int, default=0, help="context size for a local model / Ollama num_ctx (default 8192 / 16384)")
    ap.add_argument("--gpu-layers", type=int, default=int(cfg.get("gpu_layers", 0)), help="layers to offload to the GPU (local model)")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for the local model")
    ap.add_argument("--tool-mode", choices=["auto", "native", "text"], default="auto", help="native function calling or prompt-based")
    ap.add_argument("--vision-model", default=cfg.get("vision_model") or "")
    ap.add_argument("--base-url", default=getenv("BASE_URL") or cfg.get("base_url") or DEFAULT_BASE_URL)
    ap.add_argument("--cwd", default=".", help="project directory")
    ap.add_argument("--yolo", action="store_true", help="auto-approve edits and commands (destructive ones still ask)")
    ap.add_argument("--plan", action="store_true", help="start in read-only plan mode")
    ap.add_argument("--tools", choices=["all", "core"], default=None, help="'core' disables optional tool groups (default: all, but core for small-context local models)")
    ap.add_argument("--no-mcp", action="store_true", help="do not start MCP servers")
    ap.add_argument("--no-vscode-mcp", action="store_true", help="do not import MCP servers from VS Code / Cursor / Claude Desktop")
    ap.add_argument("--mcp-config", default="", help="extra MCP config file (VS Code or .mcp.json format)")
    ap.add_argument("--trust-project", action="store_true", help="load project-level tools/MCP config without asking")
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=None, help="sampling temperature (default: per-model profile, else 0.3)")
    ap.add_argument("--context-window", type=int, default=0, help="token budget before auto-compaction (default 100000; n-ctx for local)")
    ap.add_argument("--max-steps", type=int, default=0, help="max tool-use iterations per message (default 40, 200 in auto mode)")
    ap.add_argument("--ui", choices=["classic", "modern"], default=cfg.get("ui", ""), help="terminal look (default: classic)")
    ap.add_argument("--output-format", choices=["text", "json", "stream-json"], default="text", help="with -p: machine-readable output for scripts/CI")
    ap.add_argument("--no-preload", dest="preload", action="store_false", default=bool(cfg.get("preload", True)),
                    help="do not load a local model into memory before the first message")
    ap.add_argument("--show-reasoning", action="store_true")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--scan", nargs="?", const=".", default=None, metavar="TARGET",
                    help="scan a repository (folder, GitHub URL or owner/repo) and print the report - no model or API key needed")
    ap.add_argument("--update", action="store_true", help="check GitHub for a newer Harbor CLI and install it now (opens the installer)")
    ap.add_argument("--check-update", action="store_true", help="print whether a newer version exists and exit")
    ap.add_argument("--no-update-check", action="store_true", help="do not look for updates at start-up")
    ap.add_argument("--doctor", action="store_true", help="check optional dependencies and exit")
    ap.add_argument("--version", action="version", version="harbor " + VERSION)
    args = ap.parse_args(argv)

    cwd = Path(args.cwd).expanduser().resolve()
    if not cwd.is_dir():
        print("Not a directory: %s" % cwd, file=sys.stderr)
        return 2
    os.chdir(cwd)
    gitfinder.CONFIG.update(cfg)
    ui.cwd = cwd
    if migrated:
        ui.info("Migrated your settings from ~/.nimcode to ~/.harbor (the old folder was left untouched).")
    if args.ui:
        ui.style = args.ui
    if args.vision_model:
        cfg["vision_model"] = args.vision_model

    if args.scan is not None:
        from .tools.scan import format_report, resolve_and_scan
        try:
            report = resolve_and_scan(args.scan or ".", cwd)
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            print("Error: %s" % e, file=sys.stderr)
            return 2
        print(json.dumps(report, indent=2, default=str) if args.output_format == "json" else format_report(report))
        return 0

    if args.doctor:
        for ln in doctor_lines(None):
            print(ln)
        return 0

    if args.check_update:
        info, status = updater.check(cfg, force=True)
        print("Harbor CLI %s  (repo: %s, channel: %s)" % (VERSION, cfg.get("update_repo") or REPO, cfg.get("update_channel", "main")))
        print({"available": "Update available: %s  (install it with: harbor --update)" % (info or {}).get("version"),
               "current": "You are up to date.", "unreachable": "Could not reach GitHub.", "bad-repo": "update_repo must be a github.com owner/repo.",
               "skipped": "Up to date (a newer version was skipped)."}.get(status, status))
        return 0
    if args.update:
        updater.startup_update(cfg, force=True, report=print)       # restarts Harbor itself when it installed something
        return 0
    prev_version = getenv("JUST_UPDATED")
    os.environ.pop("HARBOR_JUST_UPDATED", None)
    if (sys.stdin.isatty() and sys.stdout.isatty() and not args.print_mode and not args.no_update_check and not args.list_models
            and not getenv("NO_UPDATE") and not getenv("NO_UPDATE_ONCE")):
        updater.startup_update(cfg, report=ui.info)
    if prev_version:
        ui.info("Updated Harbor CLI %s -> %s. (the previous version is backed up in ~/.harbor/backups)" % (prev_version, VERSION))

    try:
        client, model_name, backend_desc, n_ctx = choose_backend(args, cfg, main_dir)
    except NIMError as e:
        print("Error: %s" % e.message, file=sys.stderr)
        return 2
    if args.list_models:
        try:
            print("\n".join(client.list_models()))
            return 0
        except NIMError as e:
            print("Error: %s" % e.message, file=sys.stderr)
            return 1

    if args.tools is None:
        args.tools = cfg.get("tools") or ("core" if (n_ctx and n_ctx < 32768) else "all")
        if args.tools == "core" and n_ctx and not args.print_mode:
            ui.info("Small context window (%d tokens): using the core tool set. Use --tools all (or --n-ctx 32768+) for every tool." % n_ctx)
    profile = profile_for(model_name)
    apply_profile(client, profile)
    temperature = args.temperature if args.temperature is not None else (profile.temperature if profile.temperature is not None else 0.3)
    reg, mcp_clients, custom, agents, skills = build_registry(cwd, cfg, args)
    interactive = sys.stdin.isatty() and not args.print_mode
    ctx = Context(cwd, cfg, client)
    ctx.interactive = interactive or (sys.stdin.isatty() and sys.stdout.isatty())
    ctx.agents, ctx.skills = agents, skills
    perms = Permissions(args.yolo)
    perms.set_rules(cfg.get("permissions"))
    context_window = args.context_window or (n_ctx or 100000)
    max_tokens = args.max_tokens or (min(4096, max(512, (n_ctx or 0) // 4)) if n_ctx else 8192)
    max_steps = args.max_steps or (200 if args.auto else 40)
    agent = Agent(ctx, reg, perms, model_name, max_tokens, temperature, context_window, max_steps, args.show_reasoning,
                  auto=args.auto, verify_cmd=args.verify, max_minutes=args.max_minutes, guidance=profile.guidance)
    ctx.agent = agent
    start_mode = ("plan" if args.plan else "autobypass" if (args.auto and args.yolo) else "auto" if args.auto else
                  "bypass" if args.yolo else cfg.get("default_mode", "manual"))
    set_mode(agent, start_mode if start_mode in MODE_INFO else "manual")
    atexit.register(lambda: [p.proc.poll() is None and __import__("harbor.tools.shell", fromlist=["kill_tree"]).kill_tree(p.proc) for p in ctx.procs.values()])

    session_file = session_path(cwd, time.strftime("%Y%m%d-%H%M%S"))
    if args.cont:
        last = latest_session(cwd)
        if last:
            try:
                data = json.loads(last.read_text(encoding="utf-8"))
                agent.messages += data.get("messages", [])
                session_file = last
                ui.info("Resumed session %s (%d messages)" % (last.name, len(data.get("messages", []))))
            except (OSError, ValueError):
                ui.warn("Could not read the previous session; starting fresh.")
        else:
            ui.info("No previous session for this directory; starting fresh.")

    prompt = " ".join(args.prompt).strip()
    if args.print_mode:
        if not prompt and not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        if not prompt:
            print("Error: -p requires a prompt.", file=sys.stderr)
            return 2
        fmt = args.output_format
        events: list = []
        if fmt != "text":
            ui.quiet = True                       # silence the human UI; emit JSON instead

            def on_event(ev):
                events.append(ev)
                if fmt == "stream-json":
                    sys.stdout.write(json.dumps(ev) + "\n")
                    sys.stdout.flush()
            agent.events.append(on_event)
        final, rc, err = "", 0, ""
        try:
            final = agent.run_turn(prompt)
        except NIMError as e:
            rc, err = 1, "NIM API error (%s): %s" % (e.status, e.message)
        except KeyboardInterrupt:
            rc, err = 130, "interrupted"
        if fmt == "text":
            if err:
                print(err, file=sys.stderr)
            return rc
        changed = [str(p) for p, _ in (ctx.change_history[-1] if ctx.change_history else [])]
        result = {"type": "result", "subtype": "success" if rc == 0 else "error", "is_error": rc != 0, "result": final or "",
                  "error": err, "model": model_name, "tool_calls": sum(1 for e in events if e["type"] == "tool_use"),
                  "files_changed": changed, "usage": {"requests": agent.requests, "prompt_tokens": agent.prompt_tokens,
                                                      "completion_tokens": agent.completion_tokens}}
        print(json.dumps(result))
        return rc
    state = {"cfg": cfg, "args": args, "mcp": mcp_clients, "custom": custom}
    repl(agent, state, prompt, session_file)
    return 0
