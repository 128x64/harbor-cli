"""Shell, background processes and git. Windows: PowerShell (pwsh/powershell) or cmd; POSIX: bash/sh."""
from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

from ..core import Registry, shell_is_dangerous
from ..gitfinder import find_git, tool_env
from ..util import IS_WIN, bold, cyan, decode_bytes, truncate


def shell_name() -> str:
    if IS_WIN:
        if os.environ.get("NIMCODE_SHELL", "").lower() == "cmd":
            return "cmd"
        return "powershell" if (shutil.which("pwsh") or shutil.which("powershell")) else "cmd"
    return "bash" if shutil.which("bash") else "sh"


def shell_argv(command: str) -> list:
    name = shell_name()
    if IS_WIN:
        if name == "powershell":
            exe = shutil.which("pwsh") or shutil.which("powershell")
            prefix = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; $ProgressPreference='SilentlyContinue'; "
            return [exe, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", prefix + command]
        return ["cmd.exe", "/d", "/s", "/c", command]
    return [shutil.which("bash") or "/bin/sh", "-c", command]


def popen(argv, cwd, env=None):
    env = env if env is not None else tool_env()
    kw = {}
    if IS_WIN:
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kw["start_new_session"] = True
    return subprocess.Popen(argv, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=env, **kw)


def kill_tree(proc) -> None:
    try:
        if IS_WIN:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=15)
        else:
            os.killpg(proc.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_command_rc(argv, cwd, timeout=120, env=None):
    """Run a process; returns (exit_code, output_text). exit_code is None on timeout/start failure."""
    timeout = max(1, min(int(timeout or 120), 900))
    try:
        proc = popen(argv, cwd, env)
    except FileNotFoundError as e:
        return None, "Error: command not found: %s" % e
    except OSError as e:
        return None, "Error: could not start process: %s" % e
    try:
        out, _ = proc.communicate(timeout=timeout)
        rc, note = proc.returncode, ("" if proc.returncode == 0 else "\n[exit code %d]" % proc.returncode)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        out, _ = proc.communicate()
        rc, note = None, "\n[timed out after %ds and was killed]" % timeout
    except KeyboardInterrupt:
        kill_tree(proc)
        raise
    return rc, truncate(decode_bytes(out or b"").rstrip() + note) or "(no output)"


def run_command(argv, cwd, timeout=120, env=None) -> str:
    return run_command_rc(argv, cwd, timeout, env)[1]


def split_args(s: str) -> list:
    parts = shlex.split(s, posix=not IS_WIN)
    return [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'" else p for p in parts]


READONLY_GIT = {"status", "diff", "log", "show", "branch", "remote", "blame", "describe", "rev-parse", "ls-files",
                "shortlog", "tag", "config", "grep", "stash list", "ls-tree", "cat-file", "reflog", "whoami"}


def _git_is_mutating(a: dict) -> bool:
    try:
        toks = split_args(a.get("args", ""))
    except ValueError:
        return True
    toks = [t for t in toks if not t.startswith("-")] if toks and toks[0].startswith("-") else toks
    if not toks:
        return False
    sub = toks[0]
    if sub in ("branch", "tag", "remote", "config") and len(toks) > 1 and not toks[1].startswith("-"):
        return sub != "remote" or toks[1] not in ("-v", "show", "get-url")
    if sub == "stash":
        return not (len(toks) > 1 and toks[1] == "list")
    return sub not in READONLY_GIT


def dulwich_git(cwd, args: str) -> str:
    """Best-effort pure-Python git (pip install dulwich) used only when no git executable can be found or started."""
    try:
        from dulwich import porcelain
    except ImportError:
        return ("Error: git was not found. Install Git for Windows or GitHub Desktop (NimCode finds its bundled git), set "
                "NIMCODE_GIT, or `pip install dulwich` for a limited pure-Python fallback.")
    import io
    toks = split_args(args) or ["status"]
    sub, rest, repo = toks[0], toks[1:], str(cwd)
    try:
        if sub == "init":
            porcelain.init(repo)
            return "Initialized empty repository in %s" % repo
        if sub == "clone" and rest:
            porcelain.clone(rest[0], rest[1] if len(rest) > 1 else None)
            return "Cloned %s" % rest[0]
        if sub == "status":
            st = porcelain.status(repo)
            dec = lambda items: [i.decode("utf-8", "replace") if isinstance(i, bytes) else str(i) for i in items]
            lines = ["staged %s: %s" % (k, ", ".join(dec(v))) for k, v in st.staged.items() if v]
            lines += ["modified (unstaged): " + ", ".join(dec(st.unstaged))] if st.unstaged else []
            lines += ["untracked: " + ", ".join(dec(st.untracked))] if st.untracked else []
            return "\n".join(lines) or "nothing to commit, working tree clean"
        if sub == "log":
            n = 10
            for i, t in enumerate(rest):
                if t.startswith("-n") and t[2:].isdigit():
                    n = int(t[2:])
                elif t.lstrip("-").isdigit():
                    n = int(t.lstrip("-"))
            buf = io.StringIO()
            porcelain.log(repo, max_entries=n, outstream=buf)
            return buf.getvalue() or "(no commits)"
        if sub == "diff":
            buf = io.BytesIO()
            porcelain.diff(repo, outstream=buf)
            return decode_bytes(buf.getvalue()) or "(no changes)"
        if sub == "add":
            st = porcelain.status(repo)
            wanted = [p for p in rest if not p.startswith("-")]
            if not wanted or wanted in (["."], ["-A"]):
                paths = [p.decode() if isinstance(p, bytes) else p for p in list(st.unstaged) + list(st.untracked)]
                paths = [os.path.join(repo, p) for p in paths]
            else:
                paths = [os.path.join(repo, p) for p in wanted]
            porcelain.add(repo, paths=paths)
            return "Staged %d path(s)" % len(paths)
        if sub == "commit":
            msg = ""
            for i, t in enumerate(rest):
                if t in ("-m", "--message") and i + 1 < len(rest):
                    msg = rest[i + 1]
            if not msg:
                return "Error: commit needs -m \"message\""
            h = porcelain.commit(repo, message=msg.encode("utf-8"))
            return "Committed %s" % (h.decode() if isinstance(h, bytes) else h)
        if sub == "branch":
            names = porcelain.branch_list(repo)
            return "\n".join(n.decode() if isinstance(n, bytes) else str(n) for n in names)
        if sub in ("fetch", "pull", "push"):
            getattr(porcelain, sub)(repo, rest[0] if rest else "origin")
            return "%s done" % sub
    except Exception as e:  # noqa: BLE001
        return "Error (pure-Python git fallback): %s: %s" % (type(e).__name__, e)
    return "Error: the pure-Python git fallback does not support 'git %s'. Install Git for Windows or GitHub Desktop." % sub


def register(reg: Registry) -> None:
    sh = shell_name()
    desc = ("Run a shell command in the project directory and return combined stdout+stderr. Shell on this machine: %s. "
            "Each call is a fresh non-interactive shell (cd does not persist; use the cwd of the project). Do not run "
            "commands that wait for input; use start_process for servers/watchers." % (
                "PowerShell (use PowerShell syntax)" if sh == "powershell" else
                "cmd.exe (use cmd syntax)" if sh == "cmd" else sh))

    @reg.tool("shell", desc, {"command": {"type": "string"},
                              "timeout": {"type": "integer", "description": "seconds (default 120, max 900)"}},
              ["command"], group="core", ask=True, danger=lambda a: shell_is_dangerous(str(a.get("command", ""))),
              summary_key="command",
              preview=lambda ctx, a: str(a.get("command", "")))
    def shell(ctx, command, timeout=120):
        return run_command(shell_argv(command), ctx.cwd, timeout)

    # ---- background processes ---------------------------------------------
    class Proc:
        def __init__(self, proc, command):
            self.proc, self.command, self.buf, self.read_pos, self.started = proc, command, bytearray(), 0, time.time()
            self.lock = threading.Lock()
            threading.Thread(target=self._pump, daemon=True).start()

        def _pump(self):
            try:
                while True:
                    chunk = self.proc.stdout.read1(4096) if hasattr(self.proc.stdout, "read1") else self.proc.stdout.read(1)
                    if not chunk:
                        break
                    with self.lock:
                        self.buf += chunk
            except Exception:
                pass

        def new_output(self, all_=False) -> str:
            with self.lock:
                data = bytes(self.buf[0 if all_ else self.read_pos:])
                self.read_pos = len(self.buf)
            return decode_bytes(data)

    @reg.tool("start_process",
              "Start a long-running command in the background (dev server, watcher, build). Returns an id; read "
              "output with process_output, stop with stop_process.",
              {"command": {"type": "string"}}, ["command"], group="shell", ask=True,
              danger=lambda a: shell_is_dangerous(str(a.get("command", ""))), summary_key="command",
              preview=lambda ctx, a: str(a.get("command", "")))
    def start_process(ctx, command):
        try:
            proc = popen(shell_argv(command), ctx.cwd)
        except OSError as e:
            return "Error: could not start: %s" % e
        pid = "p%d" % (len(ctx.procs) + 1)
        ctx.procs[pid] = Proc(proc, command)
        time.sleep(0.5)
        return "Started %s (pid %d). Initial output:\n%s" % (pid, proc.pid, truncate(ctx.procs[pid].new_output(), 3000))

    @reg.tool("process_output", "Read new output (since last read) and status of a background process.",
              {"id": {"type": "string"}, "all": {"type": "boolean", "description": "return the full log"}}, ["id"],
              group="shell", read_only=True, summary_key="id")
    def process_output(ctx, id, all=False):
        p = ctx.procs.get(id)
        if not p:
            return "Error: no such process '%s'. Known: %s" % (id, ", ".join(ctx.procs) or "none")
        rc = p.proc.poll()
        status = "running" if rc is None else "exited with code %s" % rc
        return "[%s: %s]\n%s" % (id, status, truncate(p.new_output(bool(all)), 15000) or "(no new output)")

    @reg.tool("stop_process", "Stop a background process (kills its whole process tree).",
              {"id": {"type": "string"}}, ["id"], group="shell", summary_key="id")
    def stop_process(ctx, id):
        p = ctx.procs.get(id)
        if not p:
            return "Error: no such process '%s'." % id
        if p.proc.poll() is None:
            kill_tree(p.proc)
        return "Stopped %s." % id

    @reg.tool("list_processes", "List background processes started by this session.", {}, [], group="shell", read_only=True)
    def list_processes(ctx):
        if not ctx.procs:
            return "No background processes."
        return "\n".join("%s  %s  %s" % (k, "running" if v.proc.poll() is None else "exit %s" % v.proc.returncode, v.command)
                         for k, v in ctx.procs.items())

    # ---- git ------------------------------------------------------------------
    @reg.tool("git",
              "Run a git command, e.g. args='status', 'diff HEAD~1', 'log --oneline -10', 'commit -am \"msg\"'. "
              "Read-only subcommands run without asking. Never commit or push unless the user asked.",
              {"args": {"type": "string", "description": "everything after the word git"}}, ["args"],
              group="git", ask=_git_is_mutating, summary_key="args",
              danger=lambda a: bool(shell_is_dangerous("git " + str(a.get("args", "")))),
              preview=lambda ctx, a: "git " + str(a.get("args", "")))
    def git(ctx, args):
        exe = find_git()
        if not exe:
            return dulwich_git(ctx.cwd, args)
        try:
            argv = [exe, "--no-pager"] + split_args(args)
        except ValueError as e:
            return "Error: could not parse arguments: %s" % e
        env = dict(tool_env(), GIT_TERMINAL_PROMPT="0", GIT_EDITOR="true", GIT_PAGER="cat")
        rc, out = run_command_rc(argv, ctx.cwd, 120, env)
        if rc is None and out.startswith("Error: could not start"):   # blocked by policy (AppLocker etc.)
            return dulwich_git(ctx.cwd, args) + "\n[git executable could not be started: %s]" % exe
        return out
