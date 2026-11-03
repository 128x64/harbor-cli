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

from . import VERSION, gitfinder, mcp as mcplib, plugins
from .agent import Agent
from .backends import find_llama_server, find_local_model, make_local_client
from .client import DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_VISION_MODEL, NIMClient, NIMError
from .core import Context, Permissions, Registry
from .textproto import TextToolClient
from .tools import register_all
from .tools.shell import shell_name, run_command, shell_argv
from .ui import ui, G, gray, accent
from . import util
from .util import have

CONFIG_DIR = Path.home() / ".nimcode"
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
    ("/plan", "Toggle plan mode (read-only)"), ("/yolo", "Toggle auto-approve of everything"),
    ("/reasoning", "Toggle display of model reasoning"), ("/todos", "Show the task list"),
    ("/cost", "Show token usage"), ("/init", "Create a NIMCODE.md for this project"),
    ("/memory", "Show long-term memory"), ("/auto", "Run a goal autonomously until done: /auto <goal>"),
    ("/undo", "Revert the file changes of the last turn"), ("/verify", "Set/clear the verification command"),
    ("/exit", "Exit NimCode"),
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
    print("No NVIDIA API key found. Get a free one at https://build.nvidia.com (profile -> API Keys).")
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
    lines = [accent(" NimCode %s" % VERSION), "", " Slash commands:"]
    lines += ["   %-14s %s" % c for c in BUILTIN_COMMANDS]
    if custom:
        lines += ["", " Custom commands: " + ", ".join("/" + k for k in sorted(custom))]
    lines += ["", " Shortcuts:", "   ! <cmd>        run a shell command directly", "   esc            interrupt the current response",
              "   shift+tab      cycle default / accept-edits / plan mode", "   alt+enter or \\+enter   newline",
              "   up / down      history        tab  complete slash command", "   ctrl+c         clear input; twice to exit"]
    return lines


def current_mode(agent: Agent) -> str:
    return "plan" if agent.plan_mode else agent.perms.mode()


def cycle_mode(agent: Agent) -> None:
    if agent.plan_mode:
        agent.plan_mode = False
    elif agent.perms.yolo:
        agent.perms.yolo = False
    elif agent.perms.accept_edits:
        agent.perms.accept_edits, agent.plan_mode = False, True
    else:
        agent.perms.accept_edits = True
    agent.refresh_system_prompt()


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
           "Local model: %s" % (gguf or "none (no model.gguf next to nimcode.py -> NVIDIA NIM is used)"),
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
    return ["Version:       NimCode %s" % VERSION, "Backend:       %s" % state_backend(agent),
            "Model:         %s" % agent.model, "Vision model:  %s" % (cfg.get("vision_model") or DEFAULT_VISION_MODEL),
            "Endpoint:      %s" % agent.client.base_url, "API key:       %s" % (mask(agent.client.api_key) if agent.client.api_key != "local" else "(local model)"),
            "Working dir:   %s" % agent.ctx.cwd, "Mode:          %s%s" % (current_mode(agent), " + autonomous" if agent.auto else ""),
            "Verify cmd:    %s" % (agent.verify_cmd or "(none)"), "Context window: %d tokens" % agent.context_window,
            "Shell:         %s" % shell_name(), "git:           %s" % (gitfinder.find_git() or "not found (pure-Python fallback if dulwich installed)"),
            "Tools:         %d active of %d" % (len(agent.active_tools()), len(reg.tools)),
            "MCP servers:   %s" % (", ".join("%s (%d)" % (c.name, len(c.tools)) for c in mcp_clients) or "none")]


def state_backend(agent) -> str:
    return getattr(agent.client, "backend_desc", None) or ("NVIDIA NIM" if "nvidia" in str(agent.client.base_url) else str(agent.client.base_url))


INIT_PROMPT = ("Explore this project (structure, build/test/run commands, conventions) and create a concise NIMCODE.md in the "
               "project root with guidance for future coding sessions. Keep it under 80 lines.")


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
            else:
                reg.disabled_groups.discard(parts[1])
        for g, names in sorted(reg.groups().items()):
            on = g not in reg.disabled_groups
            ui.line("  %s %-14s %s" % ("on " if on else "off", g, gray(", ".join(names) if g.startswith("mcp:") is False else "%d tools" % len(names))))
        ui.info("Use: /tools off <group>  or  /tools on <group>")
    elif cmd == "mcp":
        if not state["mcp"]:
            ui.info("No MCP servers connected. Add them to ~/.nimcode/mcp.json or .mcp.json (see README).")
        for c in state["mcp"]:
            ui.line("  %s (%d tools)" % (c.name, len(c.tools)))
            for t in c.tools[:30]:
                ui.line(gray("     %s" % t.get("name")))
    elif cmd == "plan":
        agent.plan_mode = not agent.plan_mode
        agent.refresh_system_prompt()
        ui.info("Plan mode %s." % ("ON (read-only tools)" if agent.plan_mode else "off"))
    elif cmd == "yolo":
        agent.perms.yolo = not agent.perms.yolo
        ui.warn("Auto-approve ON (destructive commands still ask)." if agent.perms.yolo else "Auto-approve off.")
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
            agent.auto = not agent.auto
            agent.refresh_system_prompt()
            ui.info("Autonomous mode %s." % ("ON (the agent keeps working until it calls finish)" if agent.auto else "off"))
        else:
            agent.auto = True
            agent.max_steps = max(agent.max_steps, 200)
            agent.refresh_system_prompt()
            state["auto_once"] = True
            return "prompt:" + arg
    elif cmd == "undo":
        restored = agent.undo()
        if restored:
            for pth in restored:
                ui.line("  restored " + ui.rel(pth))
        else:
            ui.info("Nothing to undo (only file edits made through NimCode's tools are tracked).")
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


def repl(agent: Agent, state: dict, first_prompt: str, session_file: Path) -> None:
    cfg = state["cfg"]
    notes = ["backend: " + state_backend(agent)]
    if agent.auto:
        notes.append("autonomous mode ON")
    if state["mcp"]:
        notes.append("MCP: " + ", ".join("%s (%d)" % (c.name, len(c.tools)) for c in state["mcp"]))
    if not gitfinder.find_git():
        notes.append("git not found: install Git or GitHub Desktop (see /doctor)")
    ui.welcome(VERSION, agent.model, agent.ctx.cwd, notes)
    history = load_history()
    commands = BUILTIN_COMMANDS + [("/" + k, "Custom command") for k in sorted(state["custom"])]

    def right_text():
        pct = int(100 * agent.estimate_tokens() / max(1, agent.context_window))
        short = agent.model.split("/")[-1]
        return short + (" · %d%% context" % pct if pct >= 50 else "")

    pending = first_prompt
    while True:
        if pending:
            text, pending = pending, ""
        else:
            text = ui.read_input(lambda: current_mode(agent), lambda: cycle_mode(agent), history, commands, right_text)
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
        ok = run_safely(agent, text)
        if state.pop("auto_once", False):
            agent.auto = False
            agent.refresh_system_prompt()
        if ok:
            save_session(agent, session_file)
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
    if args.tools == "core":
        reg.disabled_groups |= {g for g in reg.groups() if g != "core"}
    for g in cfg.get("disabled_groups", []):
        reg.disabled_groups.add(g)
    return reg, clients, custom


def choose_backend(args, cfg, main_dir: Path):
    """Return (client, model_name, description, n_ctx_or_None). A model.gguf next to the main script wins."""
    pkg_root = Path(__file__).resolve().parent.parent
    dirs = [main_dir, pkg_root, Path.cwd()] if args.backend != "nim" else []
    gguf = find_local_model([main_dir, pkg_root], args.gguf) if args.backend != "nim" else None
    if args.backend == "local" and not gguf:
        raise NIMError(0, "no model.gguf found next to the script (%s)%s" % (main_dir, " and --gguf path does not exist" if args.gguf else ""))
    if gguf and args.backend in ("auto", "local"):
        ui.info("Local model found: %s - using it instead of NVIDIA NIM (pass --backend nim to override)." % gguf)
        n_ctx = args.n_ctx or int(cfg.get("n_ctx", 8192))
        client, desc = make_local_client(gguf, args.local_engine, dirs, n_ctx, args.gpu_layers, args.threads, args.tool_mode, ui.info)
        client.backend_desc = "local GGUF via " + desc
        return client, gguf.name, client.backend_desc, n_ctx
    key = resolve_api_key(cfg, sys.stdin.isatty() and not args.print_mode)
    if not key:
        raise NIMError(0, "set NVIDIA_API_KEY (free key from https://build.nvidia.com), or put a model.gguf next to nimcode.py")
    if not key.startswith("nvapi-") and args.base_url == DEFAULT_BASE_URL:
        print("Warning: NVIDIA keys normally start with 'nvapi-'.", file=sys.stderr)
    client = NIMClient(key, args.base_url)
    if args.tool_mode == "text":
        client = TextToolClient(client)
    client.backend_desc = "NVIDIA NIM" if args.base_url == DEFAULT_BASE_URL else "OpenAI-compatible (%s)" % args.base_url
    return client, args.model, client.backend_desc, None


def main(argv=None, main_dir=None) -> int:
    util.init_terminal()
    main_dir = Path(main_dir).resolve() if main_dir else Path(__file__).resolve().parent.parent
    cfg = load_config()
    ap = argparse.ArgumentParser(prog="nimcode", description="Agentic coding CLI. Uses a local model.gguf next to the script if present, else NVIDIA NIM.")
    ap.add_argument("prompt", nargs="*", help="initial prompt (the session stays interactive afterwards)")
    ap.add_argument("-p", "--print", dest="print_mode", action="store_true", help="run the prompt non-interactively and exit")
    ap.add_argument("-c", "--continue", dest="cont", action="store_true", help="continue the most recent session in this directory")
    ap.add_argument("-a", "--auto", action="store_true", help="autonomous mode: keep working until the goal is done and verified (calls finish)")
    ap.add_argument("--verify", default=cfg.get("verify", ""), help="command that must pass (exit 0) before a task counts as done, e.g. 'pytest -q'")
    ap.add_argument("--max-minutes", type=float, default=0, help="time budget per request (0 = unlimited)")
    ap.add_argument("-m", "--model", default=os.environ.get("NIMCODE_MODEL") or cfg.get("model") or DEFAULT_MODEL, help="NIM model id")
    ap.add_argument("--backend", choices=["auto", "nim", "local"], default=cfg.get("backend", "auto"),
                    help="auto: local model.gguf if present, else NIM")
    ap.add_argument("--gguf", default=os.environ.get("NIMCODE_GGUF", ""), help="path to a GGUF model (overrides model.gguf discovery)")
    ap.add_argument("--local-engine", choices=["auto", "server", "python"], default="auto", help="llama-server binary or llama-cpp-python")
    ap.add_argument("--n-ctx", type=int, default=0, help="context size for the local model (default 8192)")
    ap.add_argument("--gpu-layers", type=int, default=int(cfg.get("gpu_layers", 0)), help="layers to offload to the GPU (local model)")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for the local model")
    ap.add_argument("--tool-mode", choices=["auto", "native", "text"], default="auto", help="native function calling or prompt-based")
    ap.add_argument("--vision-model", default=cfg.get("vision_model") or "")
    ap.add_argument("--base-url", default=os.environ.get("NIMCODE_BASE_URL") or cfg.get("base_url") or DEFAULT_BASE_URL)
    ap.add_argument("--cwd", default=".", help="project directory")
    ap.add_argument("--yolo", action="store_true", help="auto-approve edits and commands (destructive ones still ask)")
    ap.add_argument("--plan", action="store_true", help="start in read-only plan mode")
    ap.add_argument("--tools", choices=["all", "core"], default=cfg.get("tools", "all"), help="'core' disables optional tool groups")
    ap.add_argument("--no-mcp", action="store_true", help="do not start MCP servers")
    ap.add_argument("--no-vscode-mcp", action="store_true", help="do not import MCP servers from VS Code / Cursor / Claude Desktop")
    ap.add_argument("--mcp-config", default="", help="extra MCP config file (VS Code or .mcp.json format)")
    ap.add_argument("--trust-project", action="store_true", help="load project-level tools/MCP config without asking")
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--context-window", type=int, default=0, help="token budget before auto-compaction (default 100000; n-ctx for local)")
    ap.add_argument("--max-steps", type=int, default=0, help="max tool-use iterations per message (default 40, 200 in auto mode)")
    ap.add_argument("--show-reasoning", action="store_true")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--doctor", action="store_true", help="check optional dependencies and exit")
    ap.add_argument("--version", action="version", version="nimcode " + VERSION)
    args = ap.parse_args(argv)

    cwd = Path(args.cwd).expanduser().resolve()
    if not cwd.is_dir():
        print("Not a directory: %s" % cwd, file=sys.stderr)
        return 2
    os.chdir(cwd)
    gitfinder.CONFIG.update(cfg)
    ui.cwd = cwd
    if args.vision_model:
        cfg["vision_model"] = args.vision_model

    if args.doctor:
        for ln in doctor_lines(None):
            print(ln)
        return 0

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

    reg, mcp_clients, custom = build_registry(cwd, cfg, args)
    interactive = sys.stdin.isatty() and not args.print_mode
    ctx = Context(cwd, cfg, client)
    ctx.interactive = interactive or (sys.stdin.isatty() and sys.stdout.isatty())
    perms = Permissions(args.yolo)
    context_window = args.context_window or (n_ctx or 100000)
    max_tokens = args.max_tokens or (min(4096, max(512, (n_ctx or 0) // 4)) if n_ctx else 8192)
    max_steps = args.max_steps or (200 if args.auto else 40)
    agent = Agent(ctx, reg, perms, model_name, max_tokens, args.temperature, context_window, max_steps, args.show_reasoning,
                  auto=args.auto, verify_cmd=args.verify, max_minutes=args.max_minutes)
    ctx.agent = agent
    agent.plan_mode = args.plan
    if args.plan:
        agent.refresh_system_prompt()
    if n_ctx and args.tools == "all" and not args.print_mode:
        ui.info("Local model: %d tools may overflow a %d-token context; consider --tools core or /tools off <group>." % (len(agent.active_tools()), n_ctx)) if len(agent.active_tools()) > 30 and n_ctx <= 16384 else None
    atexit.register(lambda: [p.proc.poll() is None and __import__("nimcode.tools.shell", fromlist=["kill_tree"]).kill_tree(p.proc) for p in ctx.procs.values()])

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
        try:
            agent.run_turn(prompt)
            return 0
        except NIMError as e:
            print("NIM API error (%s): %s" % (e.status, e.message), file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            return 130
    state = {"cfg": cfg, "args": args, "mcp": mcp_clients, "custom": custom}
    repl(agent, state, prompt, session_file)
    return 0
