"""The agent loop: streaming model calls, parallel tool execution, permissions, hooks, undo checkpoints, autonomous mode
with self-verification, context compaction and sub-agents."""
from __future__ import annotations

import json
import platform
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .client import NIMError
from .core import Context, EDIT_TOOLS, Permissions, Registry, Tool
from .gitfinder import find_git
from .hooks import run_hooks
from .tools.agentic import MEMORY_FILE
from .tools.fs import plan_edit
from .textproto import StreamSplitter, _loads_lenient
from .tools.shell import run_command_rc, shell_argv, shell_name
from .ui import display_name, make_diff, plural, render_diff, ui
from .util import truncate

CONFIG_DIR = Path.home() / ".harbor"
INSTRUCTION_FILES = ("HARBOR.md", "NIMCODE.md", "CLAUDE.md", "AGENTS.md")
NO_SUB = {"task", "delegate", "ask_user", "finish", "todo_write"}


_GIT_CACHE: dict = {}


def _git_summary(cwd: Path) -> str:
    """Cached for 30 s: rebuilding the system prompt (on every mode switch) must never wait on git subprocesses."""
    hit = _GIT_CACHE.get(str(cwd))
    if hit and time.time() - hit[0] < 30:
        return hit[1]
    value = _git_summary_uncached(cwd)
    _GIT_CACHE[str(cwd)] = (time.time(), value)
    return value


def _git_summary_uncached(cwd: Path) -> str:
    exe = find_git()
    if not exe:
        return "git not found"
    try:
        def run(*a):
            return subprocess.run([exe, *a], cwd=str(cwd), capture_output=True, text=True, timeout=5).stdout.strip()
        if run("rev-parse", "--is-inside-work-tree") != "true":
            return "not a git repository"
        status = "\n".join(run("status", "--short").splitlines()[:15]) or "clean"
        return "branch %s; status:\n%s" % (run("rev-parse", "--abbrev-ref", "HEAD"), status)
    except Exception:
        return "git unavailable"


AUTO_PROMPT = (
    "\n\n# AUTONOMOUS MODE\nYou are working fully autonomously toward the user's goal; nobody will answer questions. Make reasonable "
    "assumptions and state them. Start with a short plan via todo_write and keep it updated. Work step by step with tools. After "
    "changing anything, VERIFY it for real: run the tests/build/program, read the output, fix failures and iterate until it works. "
    "Use task for parallel research and delegate for self-contained sub-tasks. Do not stop with a plain text answer: when the goal is "
    "completely achieved AND verified, call finish(summary, success=true) with what you did and the evidence. If you are truly blocked "
    "(missing credentials, impossible requirement), call finish with success=false and say what is needed.")


GROUP_LABELS = {"core": "Core: files, search, shell, planning, sub-agents", "files": "File management", "shell": "Background processes",
                "git": "Git", "web": "Web", "code": "Code & data", "media": "Vision, OCR, screen, clipboard",
                "system": "System", "browser": "Headless browser", "plugins": "Your custom tools"}


def capabilities_section(tools: list, compact: bool = False) -> str:
    """Tell the model, in words, everything it can do right now (generated from the live tool list)."""
    if not tools:
        return ""
    names = {t.name for t in tools}
    groups: dict = {}
    for t in tools:
        groups.setdefault(t.group, []).append(t.name)
    lines = []
    for g, ns in groups.items():
        if g.startswith("mcp:"):
            shown = [n.split("__", 2)[-1] for n in ns][:14]
            lines.append("- MCP server '%s' (%d tools, call as %s__...): %s%s" % (g[4:], len(ns), "mcp__" + g[4:], ", ".join(shown), " ..." if len(ns) > 14 else ""))
        elif g == "mcp":
            lines.append("- MCP resources: " + ", ".join(ns))
        else:
            lines.append("- %s: %s" % (GROUP_LABELS.get(g, g), ", ".join(ns)))
    recipes = []
    have_ = lambda *n: all(x in names for x in n)
    if have_("glob", "grep", "read_file"):
        recipes.append("Explore code: glob + grep + read_file together in ONE reply (parallel)%s." % ("; use task for broad investigations" if "task" in names else ""))
    if have_("shell", "edit_file"):
        recipes.append("Fix bugs: reproduce with shell, edit_file, re-run to confirm. Run tests/linters after every change.")
    if have_("web_search", "web_fetch"):
        recipes.append("Research: web_search, then web_fetch the 1-2 best results. Never guess an API you can look up.")
    if "read_file" in names:
        recipes.append("Documents: read_file opens PDF, DOCX, XLSX, PPTX and notebooks directly.")
    if "analyze_image" in names:
        recipes.append("Images/screens: analyze_image (and screenshot, ocr_image) lets you actually look at pictures and the UI.")
    if "python_exec" in names:
        recipes.append("Data/automation: python_exec for calculations, parsing, converting files; sqlite_query for databases.")
    if "delegate" in names:
        recipes.append("Big jobs: todo_write a plan, delegate self-contained pieces, start_process for servers/watchers.")
    if "scan_repo" in names:
        recipes.append("Repos: scan_repo(target) profiles any folder or GitHub repo (languages, entry points, deps, tests, secrets) before you dig in.")
    if "git" in names:
        recipes.append("Git: use the git tool (status/diff/log freely; commit/push only when asked).")
    head = ("\n\n# Your capabilities\nYou are fully agentic. You can act on the user's computer and the internet through the tools below; "
            "use them proactively and chain them until the job is done. Never say you cannot do something that one of these tools "
            "covers - try it. Tools available right now:\n")
    return head + "\n".join(lines) + ("" if compact or not recipes else "\n\nTypical recipes:\n" + "\n".join("- " + r for r in recipes))


MODE_TEXT = {
    "manual": "MANUAL - the user approves each file edit and shell command. If something is denied, stop and ask what they want instead.",
    "accept": "ACCEPT EDITS - file edits are pre-approved; shell commands and other actions still need approval.",
    "plan": "PLAN - read-only. Investigate and present a plan; do not change anything.",
    "auto": "AUTONOMOUS - work until the goal is done and verified, then call finish. File edits are pre-approved.",
    "bypass": "BYPASS - everything is pre-approved (destructive commands still ask). Be careful and deliberate.",
    "autobypass": "AUTONOMOUS + BYPASS - fully hands-off: finish the goal yourself and call finish; everything is pre-approved.",
}


def build_system_prompt(cwd: Path, sub: bool = False, plan_mode: bool = False, auto: bool = False, worker: bool = False,
                        tools: list | None = None, mode: str = "manual", compact: bool = False, guidance: str = "",
                        agents: dict | None = None, skills: dict | None = None) -> str:
    extra = ""
    for base in (CONFIG_DIR, cwd):
        for name in INSTRUCTION_FILES:
            f = base / name
            if f.is_file():
                try:
                    extra += "\n\n# Project instructions (%s)\n%s" % (f, f.read_text(encoding="utf-8", errors="replace")[:8000])
                except OSError:
                    pass
                break
    try:
        mem = MEMORY_FILE.read_text(encoding="utf-8")[:4000]
        if mem.strip():
            extra += "\n\n# Long-term memory (facts the user asked you to remember)\n" + mem
    except OSError:
        pass
    if agents and not sub:
        extra += "\n\n# Custom sub-agents (use task/delegate with agent=\"name\")\n" + "\n".join(
            "- %s: %s" % (n, a["description"]) for n, a in list(agents.items())[:20])
    if skills and not sub:
        extra += "\n\n# Skills (call skill(name) FIRST when a task matches)\n" + "\n".join(
            "- %s: %s" % (n, s_["description"]) for n, s_ in list(skills.items())[:20])
    sh = shell_name()
    if sub and worker:
        role = ("You are a worker sub-agent. Complete the delegated task fully (you may edit files and run commands), verify it, "
                "then reply with a concise report: what you changed (paths), how you verified it, anything left over.")
    elif sub:
        role = ("You are a research sub-agent. You have read-only tools. Investigate thoroughly, then reply with a concise, "
                "factual report (file paths with line numbers, key findings). Do not ask questions.")
    else:
        role = ("You are Harbor CLI, an interactive command-line coding agent in the style of Claude Code. You help the user with software "
                "engineering and general computer tasks by using tools in their environment.")
    plan = ("\n\n# PLAN MODE\nYou are in plan mode: only read-only tools are available. Investigate, then write a clear step-by-step plan "
            "and call exit_plan_mode(plan) so the user can approve it. Do not claim to have changed anything.") if plan_mode else ""
    modeline = "" if sub else "\n- Permission mode: " + MODE_TEXT.get(mode, MODE_TEXT["manual"])
    tone = ("# Tone and style\n"
            "- Be concise and direct: no filler, no restating the request, no long preamble or postamble. Short answers unless detail is needed.\n"
            "- Before a batch of tool calls write ONE short sentence saying what you are doing. Use markdown; no emojis unless asked.\n"
            "- Cite code as path:line.\n\n") if not compact else "# Style\n- Be concise. Cite code as path:line.\n\n"
    return (
        role + "\n\n" + tone + "# How to work\n"
        "- Be persistent: keep going until the request is fully resolved; do not hand back half-finished work.\n"
        "- Understand first: list_dir, glob, grep, read_file. Follow the project's existing conventions; check a library is already used "
        "before assuming it. Read a file before editing it.\n"
        "- Do exactly what was asked, no extra features. Prefer editing existing files to creating new ones; never create docs unless asked.\n"
        "- Prefer dedicated tools over shell (read_file not cat, grep not shell grep, edit_file not sed). Call independent read-only tools "
        "together in ONE reply (they run in parallel).\n"
        "- Verify your work (run tests, the build or the program) with shell when practical, and fix what fails before you finish.\n"
        "- For tasks with 3+ steps keep a checklist with todo_write and mark items done as you finish them. Use task for broad exploration, "
        "delegate for self-contained sub-tasks.\n"
        "- Use web_search / web_fetch for current information or documentation you are unsure about; do not guess APIs.\n"
        "- If a tool call fails, read the error and adapt. Never fabricate file contents or command output.\n"
        "- Ask (ask_user) only when truly ambiguous or the decision is the user's; otherwise proceed with sensible assumptions.\n"
        "- Commands run in a fresh non-interactive %s each time (cd does not persist). Use start_process for servers.\n"
        "- Do not commit, push, delete data or send requests with secrets unless asked. Never reveal API keys. Decline clearly malicious requests.\n"
        "- Finish with a brief summary: what changed, where, how you verified it."
        "%s%s\n\n"
        "# Environment\n- Working directory: %s\n- OS: %s\n- Shell: %s\n- Today: %s\n- Git: %s%s%s%s"
        % (sh, modeline, capabilities_section(tools or [], compact), cwd, platform.platform(), sh, date.today().isoformat(),
           _git_summary(cwd), extra, plan, (AUTO_PROMPT if auto else "") + guidance)
    )


@dataclass
class Prepared:
    name: str
    tool: Tool | None = None
    args: dict = field(default_factory=dict)
    raw: str = "{}"
    error: str | None = None
    summary: str = ""
    label: str | None = None
    edit: tuple | None = None
    decision: str = ""     # permission rule outcome: allow | ask | deny | ""
    rule: str = ""

    @property
    def parallel_ok(self) -> bool:
        return (self.tool is not None and not self.error and self.tool.read_only and not self.tool.needs_ask(self.args)
                and self.decision != "ask" and self.tool.name not in ("ask_user", "exit_plan_mode"))


class Agent:
    def __init__(self, ctx: Context, registry: Registry, perms: Permissions, model: str, max_tokens=8192,
                 temperature=0.3, context_window=100000, max_steps=40, show_reasoning=False, sub=False, worker=False,
                 auto=False, verify_cmd="", max_minutes=0, guidance="", allowed_tools=None):
        self.ctx, self.registry, self.perms = ctx, registry, perms
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.context_window, self.max_steps, self.sub, self.worker = context_window, max_steps, sub, worker
        self.auto, self.verify_cmd, self.max_minutes, self.guidance = auto, verify_cmd, max_minutes, guidance
        self.max_nudges, self.max_verify = 3, 5
        ui.show_reasoning = show_reasoning or ui.show_reasoning
        self.plan_mode = False
        self.prompt_tokens = self.completion_tokens = self.requests = 0
        self.allowed_tools = allowed_tools
        self.events: list = []           # callables(event_dict) - used for --output-format json/stream-json
        self._recent: list = []
        self._lock = threading.Lock()
        self._rejected = False
        self._finished = False
        self._finish_ok = True
        self._finish_summary = ""
        self.messages = [{"role": "system", "content": self._system()}]
        if not sub:
            ctx.client.on_retry = lambda m: ui.info("  " + m)
            ctx.client.on_chars = ui.add_tokens        # live token counter: every streamed chunk (text, reasoning, tool args)

    # ---- helpers -------------------------------------------------------------
    @property
    def client(self):
        return self.ctx.client

    def mode_key(self) -> str:
        if self.plan_mode:
            return "plan"
        if self.auto and self.perms.yolo:
            return "autobypass"
        if self.perms.yolo:
            return "bypass"
        if self.auto:
            return "auto"
        return "accept" if self.perms.accept_edits else "manual"

    def _system(self) -> str:
        return build_system_prompt(self.ctx.cwd, self.sub, self.plan_mode, self.auto, self.worker, self.active_tools(),
                                   self.mode_key(), self.context_window <= 16384, self.guidance,
                                   getattr(self.ctx, "agents", None), getattr(self.ctx, "skills", None))

    def set_mode(self, key: str) -> None:
        """manual | accept | plan | auto | bypass | autobypass"""
        self.plan_mode = key == "plan"
        self.auto = key in ("auto", "autobypass")
        self.perms.yolo = key in ("bypass", "autobypass")
        self.perms.accept_edits = key in ("accept", "auto")
        self.refresh_system_prompt()

    def emit(self, event: dict) -> None:
        for fn in self.events:
            try:
                fn(event)
            except Exception:
                pass

    def exit_plan_mode(self, plan: str) -> str:
        if not self.plan_mode:
            return "Not in plan mode; nothing to approve."
        if not ui.interactive:
            return "Plan shown. This session is non-interactive, so it stays in plan mode; the user must approve and switch modes."
        lines = plan.strip().splitlines()
        body = lines[:24] + (["... +%d more lines" % (len(lines) - 24)] if len(lines) > 24 else [])
        idx = ui.confirm("Ready to code?", body, "Would you like to proceed?",
                         ["Yes, and auto-accept edits", "Yes, and manually approve each edit", "No, keep planning"])
        if idx == 0:
            self.set_mode("accept")
            return "The user approved the plan and enabled auto-accept for edits. Plan mode is off: implement it now."
        if idx == 1:
            self.set_mode("manual")
            return "The user approved the plan; they will approve each edit. Plan mode is off: implement it now."
        return "The user wants to keep planning. Ask what to change (ask_user) or revise the plan, then call exit_plan_mode again."

    def refresh_system_prompt(self):
        self.messages[0] = {"role": "system", "content": self._system()}

    def reset(self):
        self.messages = [self.messages[0]]
        self.ctx.todos = []
        self.refresh_system_prompt()

    def estimate_tokens(self) -> int:
        return int(sum(len(json.dumps(m, ensure_ascii=False)) for m in self.messages) / 3.5)

    def active_tools(self):
        tools = self.registry.active(self.plan_mode)
        drop = {"ask_user"} if self.auto else {"finish"}      # autonomous: nobody answers questions; otherwise `finish` is meaningless
        if not self.plan_mode or self.sub:
            drop.add("exit_plan_mode")
        if not getattr(self.ctx, "skills", None):
            drop.add("skill")
        tools = [t for t in tools if t.name not in drop]
        if self.sub:
            if self.allowed_tools:
                tools = [t for t in tools if t.name in self.allowed_tools and t.name not in NO_SUB]
            else:
                tools = [t for t in tools if t.name not in NO_SUB and (self.worker or t.read_only)]
        return tools

    def compact(self, quiet=False) -> None:
        if len(self.messages) <= 2:
            if not quiet:
                ui.info("Nothing to compact.")
            return
        lines = []
        for m in self.messages[1:]:
            body = m.get("content") or ""
            body = (body if isinstance(body, str) else json.dumps(body))[:3000]
            for tc in m.get("tool_calls") or []:
                body += "\n[called %s %s]" % (tc["function"]["name"], tc["function"]["arguments"][:500])
            lines.append("%s: %s" % (m["role"].upper(), body))
        transcript = "\n\n".join(lines)[-60000:]
        if not quiet:
            ui.info("Compacting conversation...")
        resp = self.client.chat_stream(
            self.model,
            [{"role": "system", "content": "You write dense, accurate summaries of coding sessions."},
             {"role": "user", "content": "Summarize this agent conversation so work can continue from the summary alone: the user's "
                                         "goals, files created/changed (paths), key decisions, commands run and results, errors, and "
                                         "remaining TODOs.\n\n" + transcript[-int(self.context_window * 2.5):]}],
            tools=None, max_tokens=min(2048, max(256, self.context_window // 4)), temperature=0.2)
        summary = resp["content"].strip() or "(no summary produced)"
        self.messages = [self.messages[0],
                         {"role": "user", "content": "[Summary of the earlier conversation]\n" + summary},
                         {"role": "assistant", "content": "Understood. I have the context from that summary and will continue."}]
        if not quiet:
            ui.info("Compacted (~%d tokens now)." % self.estimate_tokens())

    # ---- model call ----------------------------------------------------------
    def call_model(self, _retry=True) -> dict:
        specs = [t.spec() for t in self.active_tools()]
        sp = StreamSplitter(ui.assistant_text, ui.reasoning, only_think=True)   # models that inline <think>...</think>
        try:
            resp = self.client.chat_stream(self.model, self.messages, specs, self.max_tokens, self.temperature, sp.feed, ui.reasoning)
            sp.finish()
            if "<think>" in (resp.get("content") or ""):
                resp["content"] = "".join(sp.text).strip()
        except NIMError as e:
            ui.assistant_end()
            if e.status == 400 and _retry and re.search(r"context|too long|maximum", e.message, re.I):
                ui.info("  context too large; compacting and retrying...")
                self.compact(quiet=True)
                return self.call_model(False)
            raise
        except BaseException:
            ui.assistant_end()
            raise
        ui.assistant_end()
        if resp.get("content") and not self.sub:
            self.emit({"type": "assistant", "text": resp["content"]})
        with self._lock:
            self.requests += 1
            u = resp.get("usage") or {}
            if u:
                self.prompt_tokens += u.get("prompt_tokens", 0)
                self.completion_tokens += u.get("completion_tokens", 0)
            else:
                self.prompt_tokens += self.estimate_tokens()
                self.completion_tokens += len(resp["content"]) // 4
        return resp

    # ---- tools: prepare -> (permission) -> run -> render -----------------------
    def find_tool(self, name: str):
        t = self.registry.get(name)
        if t:
            return t
        n = name.split(".")[-1].strip()
        t = self.registry.get(n)
        if t:
            return t
        return {k.lower(): v for k, v in self.registry.tools.items()}.get(n.lower())

    def prepare(self, name: str, raw_args: str) -> Prepared:
        prep = Prepared(name=name, raw=raw_args)
        tool = self.find_tool(name)
        active = {t.name for t in self.active_tools()}
        if tool is None or tool.name not in active:
            why = " (not available in %s)" % ("plan mode" if self.plan_mode else "this context") if tool else ""
            prep.error = "Error: unknown or unavailable tool '%s'%s. Available: %s" % (name, why, ", ".join(sorted(active)))
            return prep
        prep.tool = tool
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except ValueError:
            prep.error = "Error: tool arguments were not valid JSON. Resend the call with a valid JSON object."
            return prep
        if not isinstance(args, dict):
            prep.error = "Error: tool arguments must be a JSON object."
            return prep
        if tool.filter_args:
            args = {k: v for k, v in args.items() if k in tool.schema.get("properties", {})}
        missing = [r for r in tool.schema.get("required", []) if r not in args]
        prep.args, prep.summary = args, tool.summarize(args)
        if missing:
            prep.error = "Error: missing required argument(s): %s" % ", ".join(missing)
            return prep
        dec, rule = self.perms.match(tool, args, self.ctx.cwd)
        prep.decision, prep.rule = dec or "", rule
        if dec == "deny":
            prep.error = "Blocked by permission rule '%s'. Do not retry; ask the user how to proceed." % rule
            return prep
        if tool.name in EDIT_TOOLS:
            prep.edit = plan_edit(self.ctx, tool.name, args)
            prep.summary = ui.rel(prep.edit[0])
            prep.label = "Write" if prep.edit[1] is None and tool.name == "write_file" else "Update"
            if prep.edit[3]:
                prep.error = "Error: " + prep.edit[3]
        elif tool.name == "read_file":
            prep.summary = ui.rel(args.get("path", ""))
        return prep

    def _permission(self, prep: Prepared, force: bool = False) -> bool:
        tool, args = prep.tool, prep.args
        danger = bool(tool.danger(args)) if tool.danger else False
        if prep.edit is not None:
            p, old, new, _err = prep.edit
            body = [ui.rel(p), ""] + render_diff(make_diff(old, new)[0], 82, 30)
            title = "Create file" if old is None else "Edit file"
            question = "Do you want to %s %s?" % ("create" if old is None else "make this edit to", p.name)
        else:
            try:
                body = (tool.preview(self.ctx, args) if tool.preview else prep.summary).split("\n")[:20]
            except Exception:
                body = [prep.summary]
            title = display_name(tool.name) + (" command" if tool.name in ("shell", "start_process") else "")
            question = "Do you want to proceed?"
        return self.perms.check(tool, args, title, body, question, danger, force)

    def begin(self, prep: Prepared):
        ui.tool_header(prep.tool.name, prep.summary, prep.label)
        self.emit({"type": "tool_use", "name": prep.tool.name, "input": prep.args})

    def _snapshot(self, prep: Prepared):
        path = None
        if prep.edit is not None:
            path = prep.edit[0]
        elif prep.tool.name == "delete_path":
            path = self.ctx.path(prep.args.get("path", ""))
        if path is None:
            return
        with self._lock:
            if any(p == path for p, _ in self.ctx.turn_changes):
                return
            try:
                if path.is_file():
                    self.ctx.turn_changes.append((path, path.read_bytes()))
                elif not path.exists():
                    self.ctx.turn_changes.append((path, None))
            except OSError:
                pass

    def run(self, prep: Prepared) -> str:
        tool, args = prep.tool, prep.args
        cfg = self.ctx.config
        for rc, out in run_hooks(cfg, "PreToolUse", self.ctx.cwd, tool.name, args):
            if rc == 2:
                return "Blocked by a PreToolUse hook: " + (out or "no message")
        self._snapshot(prep)
        try:
            result = tool.func(self.ctx, **args)
        except KeyboardInterrupt:
            raise
        except TypeError as e:
            result = "Error: bad arguments for %s: %s" % (tool.name, e)
        except Exception as e:  # noqa: BLE001
            result = "Error: %s: %s" % (type(e).__name__, e)
        result = truncate(str(result))
        for rc, out in run_hooks(cfg, "PostToolUse", self.ctx.cwd, tool.name, args, result):
            if rc == 2 and out:
                result += "\n[hook output]\n" + out
        with self._lock:
            self._recent = (self._recent + [(tool.name, prep.raw, result[:200])])[-3:]
            if len(self._recent) == 3 and len(set(self._recent)) == 1:
                result += "\n[harbor: you have made this exact call 3 times with the same result; try a different approach.]"
        return result

    def end(self, prep: Prepared, result: str):
        self.emit({"type": "tool_result", "name": prep.tool.name, "output": result[:4000],
                   "is_error": result.startswith(("Error", "Blocked by"))})
        if prep.edit is not None and not result.startswith(("Error", "Blocked by")):
            p, old, new, _ = prep.edit
            rows, adds, dels = make_diff(old, new)
            if old is None:
                ui.tool_result(["Wrote %s to %s" % (plural(len((new or "").splitlines()), "line"), ui.rel(p))])
                ui.show_diff(rows[:12] + ([("sep", 0, "")] if len(rows) > 12 else []))
            else:
                ui.tool_result(["Updated %s with %s and %s" % (ui.rel(p), plural(adds, "addition"), plural(dels, "removal"))])
                ui.show_diff(rows)
        else:
            lines, is_err = ui.result_lines(prep.tool.name, prep.args, result, self.ctx.todos)
            ui.tool_result(lines, is_err)

    def execute_prepared(self, prep: Prepared) -> str:
        if prep.error:
            if prep.tool:
                ui.tool_header(prep.tool.name, prep.summary, prep.label)
                ui.tool_result([prep.error], error=True)
            return prep.error
        tool = prep.tool
        danger = bool(tool.danger(prep.args)) if tool.danger else False
        needs = tool.needs_ask(prep.args) or prep.decision == "ask"
        if needs and not (prep.decision == "allow" and not danger) and not self._permission(prep, force=prep.decision == "ask"):
            self.begin(prep)
            ui.tool_result(["User rejected this action"], error=True)
            self._rejected = True
            return ("The user rejected this tool call. STOP now and wait for the user's next instruction; "
                    "do not retry or work around it.")
        self.begin(prep)
        result = self.run(prep)
        self.end(prep, result)
        return result

    def execute_tool(self, name: str, raw_args: str) -> str:
        return self.execute_prepared(self.prepare(name, raw_args))

    def _run_calls(self, calls: list) -> None:
        """Execute one model reply's tool calls (read-only ones in parallel) and append the tool messages."""
        done = 0

        def add(tc, result):
            self.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

        try:
            if len(calls) > 1:
                preps = [None if bad else self.prepare(tc["name"], tc["args"]) for tc, _a, bad in calls]
                if all(p is not None and p.parallel_ok for p in preps):
                    with ThreadPoolExecutor(max_workers=min(8, len(preps))) as ex:
                        results = list(ex.map(self.run, preps))
                    for (tc, _a, _b), p, r in zip(calls, preps, results):
                        self.begin(p)
                        self.end(p, r)
                        add(tc, r)
                        done += 1
                    return
            for tc, _a, bad in calls:
                if self._rejected:
                    result = "Skipped: the user rejected an earlier action."
                elif bad:
                    result = "Error: tool arguments were not valid JSON. Resend the call with a valid JSON object."
                else:
                    result = self.execute_prepared(self.prepare(tc["name"], tc["args"]))
                add(tc, result)
                done += 1
        except KeyboardInterrupt:
            for tc, _a, _b in calls[done:]:
                add(tc, "Interrupted by user.")
            raise

    # ---- verification ----------------------------------------------------------
    def run_verify(self):
        ui.tool_header("shell", self.verify_cmd, "Verify")
        rc, out = run_command_rc(shell_argv(self.verify_cmd), self.ctx.cwd, 900)
        lines, _ = ui.result_lines("shell", {}, out)
        ok = rc == 0
        ui.tool_result(["Verification passed"] + lines[:3] if ok else lines, error=not ok)
        return ok, out

    # ---- main loop -----------------------------------------------------------
    def run_turn(self, user_text: str) -> str:
        if self.sub:
            return self._loop(user_text)
        self.ctx.turn_changes = []
        try:
            with ui.working():
                return self._loop(user_text)
        finally:
            if self.ctx.turn_changes:
                self.ctx.change_history.append(self.ctx.turn_changes)
            self.ctx.turn_changes = []

    def undo(self) -> list:
        if not self.ctx.change_history:
            return []
        restored = []
        for path, old in reversed(self.ctx.change_history.pop()):
            try:
                if old is None:
                    if path.exists():
                        path.unlink()
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(old)
                restored.append(path)
            except OSError:
                pass
        return restored

    def _loop(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        self._rejected = self._finished = False
        last_text, nudges, verify_rounds, stop_rounds, steps = "", 0, 0, 0, 0
        t0 = time.time()
        while steps < self.max_steps:
            steps += 1
            if self.max_minutes and time.time() - t0 > self.max_minutes * 60:
                ui.warn("  (stopped: reached the %g minute time budget)" % self.max_minutes)
                return last_text
            if self.estimate_tokens() > self.context_window * 0.75:
                ui.info("  context getting large; compacting...")
                try:
                    self.compact(quiet=True)
                except NIMError as e:
                    ui.error("  compaction failed: %s" % e.message)
            resp = self.call_model()
            last_text = resp["content"] or last_text
            msg = {"role": "assistant", "content": resp["content"] or ""}
            calls = []
            for tc in resp["tool_calls"]:
                try:
                    json.loads(tc["args"])
                    calls.append((tc, tc["args"], False))
                except ValueError:
                    fixed = _loads_lenient(tc["args"])          # trailing commas, code fences, text around the JSON...
                    if isinstance(fixed, dict):
                        tc = dict(tc, args=json.dumps(fixed))
                        calls.append((tc, tc["args"], False))
                    else:
                        calls.append((tc, "{}", True))
            if calls:
                msg["tool_calls"] = [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": a}}
                                     for tc, a, _bad in calls]
            self.messages.append(msg)

            if not calls:
                if resp["finish_reason"] == "length":
                    ui.warn("  (response cut off by max_tokens; say 'continue' or raise --max-tokens)")
                    if self.auto and nudges < self.max_nudges:
                        nudges += 1
                        self.messages.append({"role": "user", "content": "[auto] Your reply was cut off. Continue."})
                        continue
                if self.auto and not self._finished and nudges < self.max_nudges and not self.sub:
                    nudges += 1
                    ui.info("  (auto) the model stopped without calling finish; asking it to continue or finish")
                    self.messages.append({"role": "user", "content": (
                        "[auto] You replied without calling a tool or finish. If the goal is completely achieved and verified, call "
                        "finish(summary, success). Otherwise keep working on the next step with tools.")})
                    continue
                if not self.sub:
                    if self.verify_cmd and self.ctx.turn_changes and verify_rounds < self.max_verify:
                        ok, out = self.run_verify()
                        if not ok:
                            verify_rounds += 1
                            self.messages.append({"role": "user", "content": "Verification command `%s` failed. Output:\n%s\nFix the problem and continue." % (self.verify_cmd, out[-4000:])})
                            continue
                    if stop_rounds < 3:
                        fb = [o for rc, o in run_hooks(self.ctx.config, "Stop", self.ctx.cwd) if rc == 2 and o]
                        if fb:
                            stop_rounds += 1
                            self.messages.append({"role": "user", "content": "[Stop hook feedback]\n" + "\n".join(fb)[:4000]})
                            continue
                if not resp["content"].strip():
                    ui.warn("  (model returned an empty reply)")
                return resp["content"]

            nudges = 0
            self._run_calls(calls)
            if self._rejected:
                return last_text
            if self._finished:
                if self.verify_cmd and verify_rounds < self.max_verify:
                    ok, out = self.run_verify()
                    if not ok:
                        verify_rounds += 1
                        self._finished = False
                        self.messages.append({"role": "user", "content": "Verification command `%s` failed, so the goal is not done. Output:\n%s\nFix the problem, then call finish again." % (self.verify_cmd, out[-4000:])})
                        continue
                return self._finish_summary or last_text
        ui.warn("  (stopped after %d steps; send a message to continue)" % self.max_steps)
        return last_text

    def run_subagent(self, description: str, prompt: str, worker: bool = False, agent: str | None = None) -> str:
        if self.sub:
            return "Error: sub-agents cannot spawn sub-agents."
        spec = None
        if agent:
            spec = self.ctx.agents.get(agent.strip().lower())
            if not spec:
                return "Error: unknown sub-agent '%s'. Available: %s" % (agent, ", ".join(sorted(self.ctx.agents)) or "none defined")
            if spec["tools"]:
                known = [self.registry.get(n) for n in spec["tools"] if self.registry.get(n)]
                worker = worker or any(not t.read_only for t in known)
        sub = Agent(self.ctx, self.registry, self.perms, (spec or {}).get("model") or self.model, self.max_tokens, self.temperature,
                    self.context_window, min(self.max_steps, 40 if worker else 25), False, sub=True, worker=worker,
                    guidance=self.guidance + (("\n\n# Your role (%s)\n%s" % (agent, spec["prompt"])) if spec else ""),
                    allowed_tools=(spec or {}).get("tools"))
        ui.sub = True
        try:
            text = sub.run_turn(prompt)
        except NIMError as e:
            return "Error: sub-agent failed: %s" % e.message
        finally:
            ui.sub = False
            with self._lock:
                self.requests += sub.requests
                self.prompt_tokens += sub.prompt_tokens
                self.completion_tokens += sub.completion_tokens
        return text.strip() or "(sub-agent produced no report)"
