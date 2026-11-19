"""Repository scanner: structure, languages, entry points, dependencies, tests, CI, licence, and security / hygiene findings.

Targets: a local folder, a GitHub URL (https://github.com/owner/repo[/tree/branch]) or `owner/repo`. Remote repos are fetched as
a ZIP (no git needed - works when git is blocked), falling back to `git clone --depth 1`. Set GITHUB_TOKEN for private repos.
Secrets are always REDACTED in the report. Nothing is executed from the scanned repo.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from ..core import Registry
from ..gitfinder import find_git, tool_env
from ..util import truthy
from .fs import SKIP_DIRS
from .shell import run_command_rc

LANG = {".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
        ".go": "Go", ".rs": "Rust", ".java": "Java", ".kt": "Kotlin", ".c": "C", ".h": "C", ".cpp": "C++", ".cc": "C++", ".hpp": "C++",
        ".cs": "C#", ".rb": "Ruby", ".php": "PHP", ".swift": "Swift", ".sh": "Shell", ".ps1": "PowerShell", ".bat": "Batch",
        ".html": "HTML", ".css": "CSS", ".scss": "CSS", ".sql": "SQL", ".lua": "Lua", ".scala": "Scala", ".dart": "Dart", ".vue": "Vue",
        ".md": "Markdown", ".json": "JSON", ".yml": "YAML", ".yaml": "YAML", ".toml": "TOML"}
CODE_LANGS = {v for v in LANG.values()} - {"Markdown", "JSON", "YAML", "TOML", "HTML", "CSS"}

SECRET_PATTERNS = [
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("Private key block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY")),
    ("NVIDIA API key", re.compile(r"\bnvapi-[A-Za-z0-9_\-]{20,}\b")),
    ("OpenAI-style secret key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("Hard-coded credential", re.compile(r"""(?i)\b(?:api[_-]?key|secret|passwd|password|token)\b['"]?\s*[:=]\s*['"]([^'"\s]{12,})['"]""")),
]
PLACEHOLDER = re.compile(r"(?i)example|your[_-]|xxxx|changeme|placeholder|dummy|sample|<.+>|\$\{|%\(|os\.environ|getenv|\*{4,}|^\.\.\.|test")
RISKY = [
    ("eval()/exec() calls", re.compile(r"(?<![\w.])(?:eval|exec)\s*\(")),
    ("subprocess with shell=True", re.compile(r"shell\s*=\s*True")),
    ("os.system()", re.compile(r"\bos\.system\s*\(")),
    ("pickle.load(s)", re.compile(r"\bpickle\.loads?\s*\(")),
    ("yaml.load without SafeLoader", re.compile(r"\byaml\.load\s*\((?![^)]*Loader)")),
    ("TLS verification disabled", re.compile(r"verify\s*=\s*False")),
    ("innerHTML assignment", re.compile(r"\.innerHTML\s*=")),
]
SENSITIVE_FILES = re.compile(r"(^|/)(\.env(\.(?!example|sample|template)[\w.]+)?|id_rsa|id_ed25519|credentials\.json|.*\.(pem|p12|pfx|jks|keystore)|.*\.sqlite3?)$", re.I)
ENTRY_FILES = ["main.py", "__main__.py", "app.py", "manage.py", "wsgi.py", "run.py", "cli.py", "index.js", "server.js", "app.js", "main.go",
               "src/main.rs", "src/main.ts", "src/index.ts", "Makefile", "Dockerfile", "docker-compose.yml", "docker-compose.yaml"]
CI_FILES = [".gitlab-ci.yml", "azure-pipelines.yml", "Jenkinsfile", ".circleci/config.yml", ".travis.yml", "appveyor.yml"]
GH_RX = re.compile(r"^(?:https?://)?(?:www\.)?github\.com/([\w.\-]+)/([\w.\-]+?)(?:\.git)?(?:/(?:tree|blob)/([^/\s]+)(?:/.*)?)?/?$")
SHORT_RX = re.compile(r"^([\w.\-]+)/([\w.\-]+)$")


def parse_github(target: str):
    m = GH_RX.match(target.strip())
    return (m.group(1), m.group(2), m.group(3)) if m else None


def is_remote(target: str, base: Path | None = None) -> bool:
    t = (target or "").strip()
    if not t or t in (".", ".."):
        return False
    if parse_github(t):
        return True
    if SHORT_RX.match(t) and not (Path(t).expanduser().exists() or (base and (base / t).exists())):
        return True
    return False


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    root = dest.resolve()
    for n in zf.namelist():
        if not str((root / n).resolve()).startswith(str(root)):
            raise ValueError("unsafe path in archive: %s" % n)
    zf.extractall(dest)


def fetch_github(owner: str, repo: str, branch: str, dest: Path) -> Path:
    """Download a repo snapshot into dest and return the extracted folder. ZIP first (no git needed), then git clone."""
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        url = "https://api.github.com/repos/%s/%s/zipball/%s" % (owner, repo, branch or "")
    else:
        url = "https://github.com/%s/%s/archive/%s.zip" % (owner, repo, ("refs/heads/" + branch) if branch else "HEAD")
    headers = {"User-Agent": "harbor-cli-scan"}
    if token:
        headers["Authorization"] = "Bearer " + token
    err = ""
    try:
        zpath = dest / "repo.zip"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as resp, open(zpath, "wb") as f:
            total = 0
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                total += len(chunk)
                if total > 150 * 1024 * 1024:
                    raise ValueError("repository archive is larger than 150 MB")
                f.write(chunk)
        with zipfile.ZipFile(zpath) as zf:
            _safe_extract(zf, dest / "x")
        tops = [p for p in (dest / "x").iterdir() if p.is_dir()]
        return tops[0] if len(tops) == 1 else dest / "x"
    except (urllib.error.URLError, OSError, ValueError, zipfile.BadZipFile) as e:
        err = str(e)
    exe = find_git()
    if exe:
        target = dest / "clone"
        rc, out = run_command_rc([exe, "clone", "--depth", "1"] + (["--branch", branch] if branch else []) +
                                 ["https://github.com/%s/%s.git" % (owner, repo), str(target)], dest, 180, dict(tool_env(), GIT_TERMINAL_PROMPT="0"))
        if rc == 0:
            return target
        err += " / git clone failed: " + out[-200:]
    raise RuntimeError("could not fetch %s/%s (%s). For a private repo clone it locally and scan the folder, or set GITHUB_TOKEN." % (owner, repo, err))


def _toml(path: Path) -> dict:
    try:
        import tomllib
        return tomllib.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def _read(path: Path, limit=1_000_000) -> str:
    try:
        if path.stat().st_size > limit:
            return ""
        raw = path.read_bytes()
        return "" if b"\0" in raw[:4096] else raw.decode("utf-8", "replace")
    except OSError:
        return ""


def redact(s: str) -> str:
    return (s[:4] + "…[%d chars]" % len(s)) if len(s) > 8 else "…"


def analyze(root: Path, label: str = "", secrets: bool = True, max_files: int = 20000) -> dict:
    root = Path(root).resolve()
    files, sizes, loc, langs = [], 0, Counter(), Counter()
    big, binaries = [], 0
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d not in SKIP_DIRS)
        for f in sorted(fn):
            if len(files) >= max_files:
                break
            p = Path(dp) / f
            try:
                size = p.stat().st_size
            except OSError:
                continue
            files.append(p)
            sizes += size
            if size > 5_000_000:
                big.append((p.relative_to(root).as_posix(), size))
    rel = lambda p: p.relative_to(root).as_posix()
    texts: dict = {}
    for p in files:
        lang = LANG.get(p.suffix.lower())
        text = _read(p) if (lang or p.name in ("Makefile", "Dockerfile")) else ""
        if lang and text:
            texts[p] = text
            loc[lang] += text.count("\n") + 1
            langs[lang] += 1
        elif lang is None and p.suffix.lower() in (".png", ".jpg", ".gif", ".zip", ".exe", ".dll", ".so", ".bin", ".pdf", ".jar"):
            binaries += 1
    names = {rel(p) for p in files}
    lower = {n.lower(): n for n in names}
    r: dict = {"target": label or str(root), "files": len(files), "bytes": sizes, "languages": [(k, langs[k], loc[k]) for k in loc.most_common_keys()] if False else
               [(k, langs[k], v) for k, v in loc.most_common(8)], "binary_files": binaries, "large_files": big[:5]}

    # entry points / packaging
    entries = [e for e in ENTRY_FILES if e in names]
    deps, scripts = [], {}
    pj = root / "package.json"
    if pj.is_file():
        try:
            j = json.loads(_read(pj) or "{}")
            deps += ["npm: " + k for k in list((j.get("dependencies") or {}))[:25]]
            scripts = j.get("scripts") or {}
            if j.get("main"):
                entries.append("package.json main: %s" % j["main"])
            for k in list(scripts)[:6]:
                entries.append("npm run %s" % k)
        except ValueError:
            pass
    py = _toml(root / "pyproject.toml") if (root / "pyproject.toml").is_file() else {}
    for d in (py.get("project", {}).get("dependencies") or [])[:30]:
        deps.append("py: " + re.split(r"[<>=!~\[; ]", d, 1)[0])
    for k, v in (py.get("project", {}).get("scripts") or {}).items():
        entries.append("console script: %s = %s" % (k, v))
    for n in sorted(names):
        if re.match(r"^requirements[\w.\-]*\.txt$", Path(n).name) and n.count("/") <= 1:
            for line in _read(root / n).splitlines():
                line = line.split("#")[0].strip()
                if line and not line.startswith(("-", "http")):
                    deps.append("py: " + re.split(r"[<>=!~\[; ]", line, 1)[0])
    if "go.mod" in names:
        deps += ["go: " + l.split()[0] for l in _read(root / "go.mod").splitlines() if re.match(r"^\s+[\w.\-/]+\s+v[\d.]", l)][:25]
    if "Cargo.toml" in names:
        cargo = _toml(root / "Cargo.toml")
        deps += ["rust: " + k for k in list(cargo.get("dependencies", {}))[:25]]
    seen, uniq = set(), []
    for d in deps:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    r["entry_points"], r["dependencies"] = entries[:15], uniq[:40]

    # tests / CI / docs
    test_files = [n for n in names if re.search(r"(^|/)(tests?|__tests__)/|(^|/)test_[^/]*\.py$|_test\.(py|go)$|\.(spec|test)\.[jt]sx?$", n)]
    blob = "\n".join(t[:4000] for t in texts.values())
    fw = [name for name, rx in (("pytest", r"\bpytest\b"), ("unittest", r"\bimport unittest\b"), ("jest", r"\bjest\b"), ("go test", r"\btesting\.T\b"), ("mocha", r"\bmocha\b")) if re.search(rx, blob)]
    r["tests"] = {"files": len(test_files), "frameworks": fw}
    r["ci"] = sorted([n for n in names if n.startswith(".github/workflows/") and n.endswith((".yml", ".yaml"))] + [c for c in CI_FILES if c in names])
    readme = next((lower[k] for k in lower if re.match(r"^readme(\.\w+)?$", k)), "")
    summary = ""
    if readme:
        body = [l.strip() for l in _read(root / readme).splitlines() if l.strip()]
        title = next((l.lstrip("# ").strip() for l in body if l.startswith("#")), "")
        para = next((l for l in body if not l.startswith(("#", "!", "[", "<", "```", "|", "-", "*"))), "")
        summary = (title + (" - " if title and para else "") + para)[:240]
    lic_file = next((lower[k] for k in lower if re.match(r"^(licen[sc]e|copying)(\.\w+)?$", k)), "")
    lic = ""
    if lic_file:
        t = _read(root / lic_file).lower()
        lic = next((n for n, k in (("MIT", "permission is hereby granted, free of charge"), ("Apache-2.0", "apache license"), ("GPL", "gnu general public license"),
                                   ("BSD", "redistribution and use in source and binary forms"), ("MPL", "mozilla public license"), ("Unlicense", "this is free and unencumbered")) if k in t), "custom/unknown")
    r["readme"], r["readme_summary"], r["license_file"], r["license"] = readme, summary, lic_file, lic
    top_pkgs = sorted({n.split("/")[0] for n in names if n.endswith("/__init__.py") and n.count("/") == 1})
    r["python_packages"] = top_pkgs[:10]
    launcher = next((n for n in ("main.py", "run.py", "app.py") if n in names), "")
    r["harbor_cli"] = bool(launcher and re.search(r"from\s+(harbor|nimcode)\.cli\s+import\s+main", _read(root / launcher)))

    # findings
    findings = {"secrets": [], "sensitive_files": [], "risky": [], "todo": 0, "notes": []}
    findings["sensitive_files"] = sorted(n for n in names if SENSITIVE_FILES.search(n))[:15]
    if secrets:
        for p, text in texts.items():
            for i, line in enumerate(text.splitlines(), 1):
                if len(line) > 600:
                    continue
                for kind, rx in SECRET_PATTERNS:
                    m = rx.search(line)
                    if not m:
                        continue
                    val = m.group(1) if (m.groups() and kind == "Hard-coded credential") else m.group(0)
                    if kind == "Hard-coded credential" and PLACEHOLDER.search(val):
                        continue
                    if kind != "Hard-coded credential" and re.search(r"(?i)example|dummy|placeholder|your[_-]", line):
                        continue
                    findings["secrets"].append({"kind": kind, "where": "%s:%d" % (rel(p), i), "preview": redact(val)})
                    break
    counts: dict = {}
    for p, text in texts.items():
        if LANG.get(p.suffix.lower()) not in CODE_LANGS:
            continue
        if re.search(r"(^|/)(tests?|__tests__)/|test_[^/]*$", rel(p)):
            continue
        for name, rx in RISKY:
            n = len(rx.findall(text))
            if n:
                counts.setdefault(name, []).append((rel(p), n))
        findings["todo"] += len(re.findall(r"\b(?:TODO|FIXME|HACK|XXX)\b", text))
    findings["risky"] = [(name, sum(n for _, n in v), sorted(v, key=lambda x: -x[1])[0][0]) for name, v in counts.items()]
    if not readme:
        findings["notes"].append("no README")
    if not lic_file:
        findings["notes"].append("no LICENSE file")
    if not test_files:
        findings["notes"].append("no tests found")
    if not r["ci"]:
        findings["notes"].append("no CI configuration")
    if ".gitignore" not in names:
        findings["notes"].append("no .gitignore")
    r["findings"] = findings
    return r


def format_report(r: dict) -> str:
    kb = lambda n: ("%.1f MB" % (n / 1048576)) if n > 1048576 else ("%d KB" % max(1, n // 1024))
    out = ["Repository scan: %s" % r["target"], "=" * 60,
           "%d files, %s%s" % (r["files"], kb(r["bytes"]), ", %d binary" % r["binary_files"] if r["binary_files"] else "")]
    if r.get("harbor_cli"):
        out.append("Harbor CLI project: yes (the launcher imports harbor.cli)")
    if r["readme_summary"]:
        out.append("About: " + r["readme_summary"])
    out.append("License: %s" % (r["license"] or "none found"))
    out += ["", "Languages (files / lines):"] + ["  %-12s %5d / %d" % x for x in r["languages"]]
    if r["python_packages"]:
        out.append("Python packages: " + ", ".join(r["python_packages"]))
    if r["entry_points"]:
        out += ["", "Entry points:"] + ["  " + e for e in r["entry_points"]]
    if r["dependencies"]:
        out += ["", "Dependencies (%d): %s" % (len(r["dependencies"]), ", ".join(d.split(": ", 1)[1] for d in r["dependencies"][:24]) + (" ..." if len(r["dependencies"]) > 24 else ""))]
    t = r["tests"]
    out += ["", "Tests: %d file(s)%s" % (t["files"], " (%s)" % ", ".join(t["frameworks"]) if t["frameworks"] else ""),
            "CI: " + (", ".join(r["ci"][:5]) if r["ci"] else "none")]
    f = r["findings"]
    out += ["", "Security & hygiene findings", "-" * 28]
    if f["secrets"]:
        out.append("Possible secrets (%d) - values redacted:" % len(f["secrets"]))
        out += ["  [%s] %s  %s" % (s["kind"], s["where"], s["preview"]) for s in f["secrets"][:15]]
        if len(f["secrets"]) > 15:
            out.append("  ... +%d more" % (len(f["secrets"]) - 15))
    else:
        out.append("Possible secrets: none detected")
    if f["sensitive_files"]:
        out.append("Sensitive-looking files in the tree: " + ", ".join(f["sensitive_files"]))
    if f["risky"]:
        out.append("Risky constructs (non-test code):")
        out += ["  %-30s %4d  (most in %s)" % x for x in f["risky"]]
    out.append("TODO/FIXME/HACK markers: %d" % f["todo"])
    if r["large_files"]:
        out.append("Large files: " + ", ".join("%s (%s)" % (n, kb(s)) for n, s in r["large_files"]))
    if f["notes"]:
        out.append("Missing: " + ", ".join(f["notes"]))
    return "\n".join(out)


def resolve_and_scan(target: str, base: Path, secrets: bool = True):
    """-> (report_dict). Fetches remote repos into a temp dir that is always removed."""
    target = (target or ".").strip()
    if is_remote(target, base):
        owner, repo, branch = parse_github(target) or (*SHORT_RX.match(target).groups(), None)
        tmp = Path(tempfile.mkdtemp(prefix="harbor_scan_"))
        try:
            root = fetch_github(owner, repo, branch, tmp)
            return analyze(root, "https://github.com/%s/%s%s" % (owner, repo, ("  (branch %s)" % branch) if branch else ""), secrets)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    p = Path(target).expanduser()
    p = p if p.is_absolute() else (base / p)
    if not p.is_dir():
        raise FileNotFoundError("not a folder: %s" % p)
    return analyze(p, str(p.resolve()), secrets)


def register(reg: Registry) -> None:
    @reg.tool("scan_repo",
              "Scan a repository (a local folder, a GitHub URL or owner/repo) and report its structure, languages, entry points, "
              "dependencies, tests, CI, licence and security/hygiene findings (possible secrets are redacted). Nothing is executed.",
              {"target": {"type": "string", "description": "folder path, https://github.com/owner/repo, or owner/repo (default '.')"},
               "secrets": {"type": "boolean", "description": "look for hard-coded secrets (default true)"}},
              [], group="code", read_only=True, summary_key="target",
              ask=lambda a: is_remote(str(a.get("target", "")), Path.cwd()))
    def scan_repo(ctx, target=".", secrets=True):
        try:
            return format_report(resolve_and_scan(target or ".", ctx.cwd, truthy(secrets)))
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            return "Error: %s" % e
