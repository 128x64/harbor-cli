"""System information tools (psutil optional)."""
from __future__ import annotations

import datetime
import os
import platform
import shutil
import sys
import time

from ..core import Registry
from ..util import IS_WIN, have, truncate
from .shell import shell_name
from ..gitfinder import find_git

SECRET_HINT = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH")


def register(reg: Registry) -> None:
    @reg.tool("system_info", "OS, CPU, memory, disk, Python version, shell and which developer tools are on PATH.",
              {}, [], group="system", read_only=True)
    def system_info(ctx):
        out = ["os: %s" % platform.platform(), "python: %s (%s)" % (sys.version.split()[0], sys.executable),
               "shell: %s" % shell_name(), "cwd: %s" % ctx.cwd, "cpu cores: %s" % os.cpu_count()]
        if have("psutil"):
            import psutil
            vm = psutil.virtual_memory()
            out.append("memory: %.1f GB total, %.1f GB available" % (vm.total / 2**30, vm.available / 2**30))
        try:
            du = shutil.disk_usage(str(ctx.cwd))
            out.append("disk (cwd): %.1f GB free of %.1f GB" % (du.free / 2**30, du.total / 2**30))
        except OSError:
            pass
        tools = ["git", "python", "pip", "node", "npm", "docker", "gcc", "make", "cmake", "go", "cargo", "java", "dotnet", "uv", "winget", "rg"]
        out.append("git: %s" % (find_git() or "not found"))
        out.append("on PATH: " + (", ".join(t for t in tools if shutil.which(t)) or "none of the usual dev tools"))
        return "\n".join(out)

    @reg.tool("system_processes", "List running OS processes sorted by memory (needs `pip install psutil`).",
              {"filter": {"type": "string", "description": "substring of process name"}}, [], group="system", read_only=True)
    def system_processes(ctx, filter=""):
        if not have("psutil"):
            return "Error: needs `pip install psutil`."
        import psutil
        rows = []
        for p in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                n = p.info["name"] or ""
                if filter.lower() in n.lower():
                    rows.append((p.info["memory_info"].rss if p.info["memory_info"] else 0, p.info["pid"], n))
            except Exception:
                pass
        rows.sort(reverse=True)
        return "\n".join("%7d  %8.1f MB  %s" % (pid, rss / 2**20, name) for rss, pid, name in rows[:40]) or "No matching processes."

    @reg.tool("datetime_now", "Current local date, time, timezone and Unix timestamp.", {}, [], group="system", read_only=True)
    def datetime_now(ctx):
        now = datetime.datetime.now().astimezone()
        return "%s (%s) | unix %d" % (now.isoformat(timespec="seconds"), time.tzname[0], int(now.timestamp()))

    @reg.tool("env_vars", "List environment variable names (values of secret-looking variables are hidden), or read one.",
              {"name": {"type": "string", "description": "optional: a specific variable"}}, [], group="system", read_only=True)
    def env_vars(ctx, name=""):
        def show(k, v):
            return "<hidden>" if any(h in k.upper() for h in SECRET_HINT) else v
        if name:
            v = os.environ.get(name)
            return "(not set)" if v is None else show(name, v)
        return truncate("\n".join("%s=%s" % (k, show(k, v)[:200]) for k, v in sorted(os.environ.items())))
