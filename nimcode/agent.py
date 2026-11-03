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
from .tools.shell import run_command_rc, shell_argv, shell_name
from .ui import display_name, make_diff, plural, render_diff, ui
from .util import truncate

CONFIG_DIR = Path.home() / ".nimcode"
INSTRUCTION_FILES = ("NIMCODE.md", "CLAUDE.md", "AGENTS.md")
NO_SUB = {"task", "delegate", "ask_user", "finish", "todo_write"}


def _git_summary(cwd: Path) -> str:
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


def build_system_prompt(cwd: Path, sub: bool = False, plan_mode: bool = False, auto: bool = False, worker: bool = False) -> str:
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
    sh = shell_name()
    if sub and worker:
        role = ("You are a worker sub-agent. Complete the delegated task fully (you may edit files and run commands), verify it, "
                "then reply with a concise report: what you changed (paths), how you verified it, anything left over.")
    elif sub:
        role = ("You are a research sub-agent. You have read-only tools. Investigate thoroughly, then reply with a concise, "
                "factual report (file paths with line numbers, key findings). Do not ask questions.")
    else:
        role = ("You are NimCode, an interactive command-line coding agent. You help the user with software engineering and "
                "general computer tasks by using tools in their environment.")
    plan = ("\n\n# PLAN MODE\nYou are in plan mode: only read-only tools are available. Investigate, then present a clear "
            "step-by-step plan. Do not claim to have changed anything.") if plan_mode else ""
    return (
        role + "\n\n# How to work\n"
        "- Be persistent: keep going until the request is fully resolved; do not hand back half-finished work.\n"
        "- Inspect before you change: list_dir, glob, grep, read_file first. Read a file before editing it.\n"
        "- Call independent read-only tools together in ONE reply (they run in parallel).\n"
        "- Prefer edit_file / multi_edit for targeted changes; write_file for new files. Keep changes minimal and in the existing style.\n"
        "- Verify your work (run tests, the build, or the program) with shell when practical, and fix what fails.\n"
        "- For tasks with 3+ steps keep a checklist with todo_write. Use task for broad exploration, delegate for self-contained sub-tasks.\n"
        "- Use web_search / web_fetch for current information or documentation you are unsure about; do not guess APIs.\n"
        "- If a tool call fails, read the error and adapt. Never fabricate file contents or command output.\n"
        "- Commands run in a fresh non-interactive %s each time (cd does not persist). Use start_process for servers.\n"
        "- Do not commit, push, delete data or send requests with secrets unless the user asked. Never reveal API keys.\n"
        "- Be concise: plain text, short code blocks only when useful, cite code as path:line. Finish with a brief summary.\n\n"
        "# Environment\n- Working directory: %s\n- OS: %s\n- Shell: %s\n- Today: %s\n- Git: %s%s%s%s"
        % (sh, cwd, platform.platform(), sh, date.today().isoformat(), _git_summary(cwd), extra, plan, AUTO_PROMPT if auto else "")
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

    @property
    def parallel_ok(self) -> bool:
        return self.tool is not None and not self.error and self.tool.read_only and not self.tool.needs_ask(self.args)


class Agent:
    def __init__(self, ctx: Context, registry: Registry, perms: Permissions, model: str, max_tokens=8192,
                 temperature=0.3, context_window=100000, max_steps=40, show_reasoning=False, sub=False, worker=False,
                 auto=False, verify_cmd="", max_minutes=0):
        self.ctx, self.registry, self.perms = ctx, registry, perms
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.context_window, self.max_steps, self.sub, self.worker = context_window, max_steps, sub, worker
        self.auto, self.verify_cmd, self.max_minutes = auto, verify_cmd, max_minutes
        self.max_nudges, self.max_verify = 3, 5
        ui.show_reasoning = show_reasoning or ui.show_reasoning
        self.plan_mode = False
        self.prompt_tokens = self.completion_tokens = self.requests = 0
        self.messages = [{"role": "system", "content": build_system_prompt(ctx.cwd, sub, False, auto, worker)}]
        self._recent: list = []
        self._lock = threading.Lock()
        self._rejected = False
        self._finished = False
        self._finish_ok = True
        self._finish_summary = ""
        if not sub:
            ctx.client.on_retry = lambda m: ui.info("  " + m)

    # ---- helpers -------------------------------------------------------------
    @property
    def client(self):
        return self.ctx.client

    def refresh_system_prompt(self):
        self.messages[0] = {"role": "system", "content": build_system_prompt(self.ctx.cwd, self.sub, self.plan_mode, self.auto, self.worker)}

    def reset(self):
        self.messages = [self.messages[0]]
        self.ctx.todos = []
        self.refresh_system_prompt()

    def estimate_tokens(self) -> int:
        return int(sum(len(json.dumps(m, ensure_ascii=False)) for m in self.messages) / 3.5)

    def active_tools(self):
        tools = self.registry.active(self.plan_mode)
        if not self.auto:
            tools = [t for t in tools if t.name != "finish"]
        else:
            tools = [t for t in tools if t.name != "ask_user"]
        if self.sub:
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
        try:
            resp = self.client.chat_stream(self.model, self.messages, specs, self.max_tokens, self.temperature,
                                           ui.assistant_text, ui.reasoning)
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
        if tool.name in EDIT_TOOLS:
            prep.edit = plan_edit(self.ctx, tool.name, args)
            prep.summary = ui.rel(prep.edit[0])
            prep.label = "Write" if prep.edit[1] is None and tool.name == "write_file" else "Update"
            if prep.edit[3]:
                prep.error = "Error: " + prep.edit[3]
        elif tool.name == "read_file":
            prep.summary = ui.rel(args.get("path", ""))
        return prep

    def _permission(self, prep: Prepared) -> bool:
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
        return self.perms.check(tool, args, title, body, question, danger)

    def begin(self, prep: Prepared):
        ui.tool_header(prep.tool.name, prep.summary, prep.label)

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
                result += "\n[nimcode: you have made this exact call 3 times with the same result; try a different approach.]"
        return result

    def end(self, prep: Prepared, result: str):
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
        if prep.tool.needs_ask(prep.args) and not self._permission(prep):
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

    def run_subagent(self, description: str, prompt: str, worker: bool = False) -> str:
        if self.sub:
            return "Error: sub-agents cannot spawn sub-agents."
        sub = Agent(self.ctx, self.registry, self.perms, self.model, self.max_tokens, self.temperature,
                    self.context_window, min(self.max_steps, 40 if worker else 25), False, sub=True, worker=worker)
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
