"""Locate a git executable. Handles machines with only GitHub Desktop installed (it bundles its own git).

Search order: $NIMCODE_GIT, config "git_path", PATH, then well-known Windows locations:
  %LOCALAPPDATA%\\GitHubDesktop\\app-<version>\\resources\\app\\git\\cmd\\git.exe   (GitHub Desktop's embedded git)
  C:\\Program Files\\Git\\cmd\\git.exe, %LOCALAPPDATA%\\Programs\\Git\\cmd\\git.exe, scoop, Visual Studio, JetBrains...
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import sys
from pathlib import Path

from .util import IS_WIN

_cache: dict = {}
CONFIG: dict = {}  # set by the CLI so git_path from ~/.nimcode/config.json is honored


def _version_key(p: str):
    m = re.search(r"app-([\d.]+)", p)
    return tuple(int(x) for x in m.group(1).split(".") if x.isdigit()) if m else ()


def _candidates() -> list:
    if not IS_WIN:
        return []
    env = os.environ
    local = env.get("LOCALAPPDATA", "")
    pf, pf86 = env.get("ProgramFiles", r"C:\Program Files"), env.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    home = str(Path.home())
    out = []
    # GitHub Desktop: newest app-* folder first
    for pat in (os.path.join(local, "GitHubDesktop", "app-*", "resources", "app", "git", "cmd", "git.exe"),
                os.path.join(local, "GitHubDesktop", "app-*", "resources", "app", "git", "mingw64", "bin", "git.exe")):
        out += sorted(glob.glob(pat), key=_version_key, reverse=True)
    out += [os.path.join(pf, "Git", "cmd", "git.exe"), os.path.join(pf86, "Git", "cmd", "git.exe"),
            os.path.join(local, "Programs", "Git", "cmd", "git.exe"),
            os.path.join(home, "scoop", "apps", "git", "current", "cmd", "git.exe")]
    out += glob.glob(os.path.join(pf, "Microsoft Visual Studio", "*", "*", "Common7", "IDE", "CommonExtensions",
                                  "Microsoft", "TeamFoundation", "Team Explorer", "Git", "cmd", "git.exe"))
    out += glob.glob(os.path.join(pf, "SmartGit", "git", "bin", "git.exe"))
    return out


def find_git(config: dict | None = None):
    """Return absolute path to git or None."""
    config = CONFIG if config is None else config
    key = config.get("git_path", "")
    if key in _cache:
        return _cache[key]
    found = None
    for cand in (os.environ.get("NIMCODE_GIT"), config.get("git_path")):
        if cand and os.path.isfile(cand):
            found = cand
            break
    if not found:
        found = shutil.which("git")
    if not found:
        for c in _candidates():
            if os.path.isfile(c):
                found = c
                break
    _cache[key] = found
    return found


def tool_env(config: dict | None = None) -> dict:
    """Environment for child processes: PATH gets git's directory prepended when git isn't already on PATH."""
    env = dict(os.environ)
    scripts = os.path.dirname(os.path.abspath(sys.executable))   # venv Scripts/bin: uv, uvx, pip-installed MCP servers
    if scripts not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = env.get("PATH", "") + os.pathsep + scripts
    g = find_git(config)
    if g and not shutil.which("git"):
        d = os.path.dirname(g)
        env["PATH"] = d + os.pathsep + env.get("PATH", "")
        # Git for Windows layouts also keep unix tools (sh, ssh) next to cmd/ in ../usr/bin and ../mingw64/bin
        root = os.path.dirname(d)
        for sub in ("mingw64/bin", "usr/bin"):
            extra = os.path.join(root, *sub.split("/"))
            if os.path.isdir(extra):
                env["PATH"] += os.pathsep + extra
    return env
