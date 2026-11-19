"""Self-update for Harbor CLI.

On start (interactive sessions only) Harbor CLI looks at its own GitHub repo. If a newer version exists it stops, opens an installer
window (Tkinter; a console installer if Tk or a display is missing), installs the update, and starts again.

Safety rules (this code downloads and installs code, so it is deliberately strict):
  * GitHub only, HTTPS only - the repo is validated as github.com/<owner>/<repo>; redirects must stay on GitHub hosts.
  * The archive is size-capped, zip-slip/symlink checked, must contain the expected files and its own VERSION must equal the
    version that was announced and be newer than the installed one.
  * Only Harbor's own files are replaced (harbor/, tests/, examples/, docs/ and a few top-level files). Your models (*.gguf),
    llama-server, .venv, config and anything else in the folder are never touched.
  * Everything replaced is backed up first (~/.harbor/backups, last 3 kept); a smoke test (`main.py --version`) must pass, otherwise the
    previous version is restored automatically.
  * Git checkouts and pip installs are never overwritten (you get a hint instead). Non-interactive runs never check.
Config (~/.harbor/config.json): "auto_update": "auto" | "ask" | "off", "update_repo": "owner/repo", "update_channel": "main" | "release".
Environment: HARBOR_NO_UPDATE=1 disables the check.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from . import REPO, VERSION
from .util import getenv

HOME_DIR = Path.home() / ".harbor"
STATE_FILE = HOME_DIR / "update.json"
BACKUP_DIR = HOME_DIR / "backups"
LOCK_FILE = HOME_DIR / "update.lock"

HOSTS = {"raw": "https://raw.githubusercontent.com", "web": "https://github.com", "api": "https://api.github.com"}
ALLOWED_HOSTS = {"raw.githubusercontent.com", "github.com", "api.github.com", "codeload.github.com", "objects.githubusercontent.com"}
MANAGED_DIRS = ("harbor", "tests", "examples", "docs")
SKIP_TOP = {".git", ".venv", "venv", "env", "__pycache__", ".harbor", ".github", ".idea", ".vscode"}
USER_FILE_RX = re.compile(r"(?i)\.(gguf|safetensors|bin|log|db|sqlite3?)$|^llama-server|^config\.json$|^\.env")
MAX_ZIP_BYTES, MAX_UNZIPPED, MAX_ENTRIES = 100 * 1024 * 1024, 300 * 1024 * 1024, 6000
EXIT = {"updated": 0, "later": 10, "skipped": 11, "failed": 20}


class UpdateError(Exception):
    pass


# ------------------------------------------------------------------------------------------- versions, repo, state
def parse_version(v: str) -> tuple:
    nums = [int(x) for x in re.findall(r"\d+", str(v))[:4]]
    return tuple(nums + [0] * (4 - len(nums)))


def is_prerelease(v: str) -> bool:
    return bool(re.search(r"(?i)(a|b|rc|alpha|beta|dev|pre)\d*$", str(v).strip()))


def newer(candidate: str, current: str = VERSION) -> bool:
    return parse_version(candidate) > parse_version(current) and not is_prerelease(candidate)


def parse_repo(value: str):
    """'owner/repo' or a github.com URL -> (owner, repo); anything else (other hosts!) -> None."""
    v = (value or "").strip()
    m = re.match(r"^(?:https://(?:www\.)?github\.com/)?([\w.\-]+)/([\w.\-]+?)(?:\.git)?/?$", v)
    return (m.group(1), m.group(2)) if m else None


def _hosts() -> dict:
    """Production talks to GitHub only. For tests a loopback base URL may be injected (never anything else)."""
    base = getenv("UPDATE_TEST_BASE")
    if base and re.match(r"^http://(127\.0\.0\.1|localhost)(:\d+)?$", base):
        return {"raw": base + "/raw", "web": base + "/web", "api": base + "/api"}
    return HOSTS


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    try:
        HOME_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        pass


def install_root() -> Path:
    return Path(__file__).resolve().parent.parent


def install_kind(root: Path | None = None) -> str:
    """folder (a downloaded copy: updatable) | git (a clone: never overwritten) | pip (site-packages) | other."""
    root = Path(root or install_root())
    parts = {p.lower() for p in root.parts}
    if "site-packages" in parts or "dist-packages" in parts:
        return "pip"
    if (root / ".git").exists():
        return "git"
    if not ((root / "main.py").is_file() and (root / "harbor" / "__init__.py").is_file()):
        return "other"
    return "folder"


# ------------------------------------------------------------------------------------------- network
def _check_host(url: str) -> None:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if host in ("127.0.0.1", "localhost") and getenv("UPDATE_TEST_BASE"):
        return
    if urllib.parse.urlparse(url).scheme != "https" or host not in ALLOWED_HOSTS:
        raise UpdateError("refusing to talk to %s (updates come from GitHub over HTTPS only)" % (host or url))


def _open(url: str, timeout: float):
    _check_host(url)
    req = urllib.request.Request(url, headers={"User-Agent": "harbor-cli-updater/" + VERSION, "Accept": "*/*"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    _check_host(resp.geturl())                       # redirects (github.com -> codeload.github.com) must stay on GitHub
    return resp


def _get(url: str, timeout: float, limit: int = 2_000_000) -> bytes:
    with _open(url, timeout) as r:
        return r.read(limit)


def fetch_latest(owner: str, repo: str, channel: str = "main", timeout: float = 3.0) -> dict:
    """Ask GitHub what the newest version is. channel "main": VERSION in harbor/__init__.py on the default branch;
    "release": the latest GitHub release tag."""
    h = _hosts()
    if channel == "release":
        data = json.loads(_get("%s/repos/%s/%s/releases/latest" % (h["api"], owner, repo), timeout).decode("utf-8", "replace"))
        tag = str(data["tag_name"])
        version = tag.lstrip("vV")
        return {"version": version, "zip_url": "%s/%s/%s/archive/refs/tags/%s.zip" % (h["web"], owner, repo, urllib.parse.quote(tag)),
                "notes": str(data.get("body") or "")[:3000], "channel": "release", "repo": "%s/%s" % (owner, repo)}
    text = _get("%s/%s/%s/HEAD/harbor/__init__.py" % (h["raw"], owner, repo), timeout).decode("utf-8", "replace")
    m = re.search(r"""VERSION\s*=\s*["']([\w.\-+]+)["']""", text)
    if not m:
        raise ValueError("no VERSION found in the repository")
    notes = ""
    try:
        log = _get("%s/%s/%s/HEAD/CHANGELOG.md" % (h["raw"], owner, repo), timeout).decode("utf-8", "replace")
        parts = re.split(r"(?m)^## ", log)
        notes = ("## " + parts[1]).strip()[:3000] if len(parts) > 1 else log[:1500]
    except (urllib.error.URLError, OSError, UpdateError, ValueError):
        pass
    return {"version": m.group(1), "zip_url": "%s/%s/%s/archive/HEAD.zip" % (h["web"], owner, repo), "notes": notes,
            "channel": "main", "repo": "%s/%s" % (owner, repo)}


def check(cfg: dict, force: bool = False, timeout: float = 3.0, now: float | None = None):
    """-> (info_dict_or_None, status) with status in: available | current | skipped | unreachable | backoff | bad-repo."""
    now = time.time() if now is None else now
    state = load_state()
    if not force and now - state.get("last_failed", 0) < 600:
        return None, "backoff"                       # offline a moment ago: do not slow every start down
    gap = float(cfg.get("update_check_minutes", 0)) * 60
    if not force and gap and now - state.get("last_check", 0) < gap:
        return None, "current"
    repo = parse_repo(str(cfg.get("update_repo") or getenv("UPDATE_REPO") or REPO))
    if not repo:
        return None, "bad-repo"
    try:
        info = fetch_latest(repo[0], repo[1], str(cfg.get("update_channel") or "main"), timeout)
    except (urllib.error.URLError, OSError, ValueError, KeyError, UpdateError):
        state["last_failed"] = now
        save_state(state)
        return None, "unreachable"
    state.update(last_check=now, latest=info["version"])
    state.pop("last_failed", None)
    save_state(state)
    if not newer(info["version"]):
        return None, "current"
    if not force and state.get("skipped") == info["version"]:
        return None, "skipped"
    return info, "available"


def skip_version(version: str) -> None:
    state = load_state()
    state["skipped"] = version
    save_state(state)


# ------------------------------------------------------------------------------------------- the update job
def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    infos = zf.infolist()
    if len(infos) > MAX_ENTRIES or sum(i.file_size for i in infos) > MAX_UNZIPPED:
        raise UpdateError("the update archive is unreasonably large")
    root = dest.resolve()
    for i in infos:
        target = (root / i.filename).resolve()
        if not str(target).startswith(str(root)) or i.filename.startswith(("/", "\\")):
            raise UpdateError("unsafe path in the update archive: %s" % i.filename)
        if (i.external_attr >> 16) & 0o170000 == 0o120000:
            raise UpdateError("symbolic links are not allowed in an update archive: %s" % i.filename)
    zf.extractall(dest)


def _sha256(p: Path) -> str:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return ""


def _acquire_lock() -> None:
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if LOCK_FILE.exists() and time.time() - LOCK_FILE.stat().st_mtime < 600:
            raise UpdateError("another Harbor CLI update is already running")
        LOCK_FILE.write_text(str(os.getpid()))
    except OSError:
        pass


def _release_lock() -> None:
    try:
        LOCK_FILE.unlink()
    except OSError:
        pass


class UpdateJob:
    """download -> verify -> back up -> replace -> (dependencies) -> smoke test -> done, with automatic rollback."""

    def __init__(self, info: dict, root: Path, progress=None, deps: bool = True, python: str | None = None):
        self.info, self.root = info, Path(root)
        self.progress = progress or (lambda msg, pct: None)
        self.deps, self.python = deps, python or sys.executable
        self.created: list = []
        self.backup: Path | None = None

    def log(self, msg: str, pct: int) -> None:
        self.progress(msg, pct)

    # -- step 1: download + verify -------------------------------------------------------------
    def download(self, tmp: Path) -> Path:
        self.log("Downloading %s ..." % self.info["zip_url"].split("/archive/")[0].replace("https://", ""), 5)
        zpath = tmp / "update.zip"
        try:
            with _open(self.info["zip_url"], 30) as r, open(zpath, "wb") as f:
                total, last = 0, 0
                length = int(r.headers.get("Content-Length") or 0)
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_ZIP_BYTES:
                        raise UpdateError("the update archive is larger than 100 MB")
                    f.write(chunk)
                    if length and total - last > (1 << 18):
                        last = total
                        self.log("Downloading ... %d%%" % (100 * total // length), 5 + int(35 * total / length))
        except (urllib.error.URLError, OSError) as e:
            raise UpdateError("download failed: %s" % e)
        self.log("Verifying the archive ...", 45)
        try:
            with zipfile.ZipFile(zpath) as zf:
                if zf.testzip() is not None:
                    raise UpdateError("the downloaded archive is corrupt")
                _safe_extract(zf, tmp / "x")
        except zipfile.BadZipFile:
            raise UpdateError("the downloaded file is not a valid zip archive")
        tops = [p for p in (tmp / "x").iterdir() if p.is_dir()]
        new_root = tops[0] if len(tops) == 1 and not any(p.is_file() for p in (tmp / "x").iterdir()) else tmp / "x"
        for need in ("main.py", "harbor/__init__.py", "harbor/cli.py"):
            if not (new_root / need).is_file():
                raise UpdateError("the update archive is missing %s - not a Harbor CLI release" % need)
        m = re.search(r"""VERSION\s*=\s*["']([\w.\-+]+)["']""", (new_root / "harbor" / "__init__.py").read_text(encoding="utf-8", errors="replace"))
        got = m.group(1) if m else ""
        if got != self.info["version"]:
            raise UpdateError("version mismatch: announced %s but the archive contains %s" % (self.info["version"], got or "?"))
        if not newer(got):
            raise UpdateError("the archive (%s) is not newer than the installed version (%s)" % (got, VERSION))
        return new_root

    # -- step 2: scan (information only) -----------------------------------------------------
    def scan(self, new_root: Path) -> None:
        try:
            from .tools.scan import analyze
            rep = analyze(new_root, "update", secrets=True)
            f = rep["findings"]
            self.log("Scanned the release: %d files, %d possible secret(s), %d sensitive file(s)" % (rep["files"], len(f["secrets"]), len(f["sensitive_files"])), 52)
        except Exception:
            pass

    # -- step 3: back up + apply ---------------------------------------------------------------
    def _managed_files(self, base: Path) -> list:
        out = []
        for d in MANAGED_DIRS:
            for p in sorted((base / d).rglob("*")) if (base / d).is_dir() else []:
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                    out.append(p.relative_to(base).as_posix())
        for p in sorted(base.iterdir()):
            if p.is_file() and p.name not in SKIP_TOP and not USER_FILE_RX.search(p.name):
                out.append(p.name)
        return out

    def make_backup(self) -> None:
        self.log("Backing up the current version ...", 58)
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        self.backup = BACKUP_DIR / ("harbor-%s-%d.zip" % (VERSION, int(time.time())))
        with zipfile.ZipFile(self.backup, "w", zipfile.ZIP_DEFLATED) as z:
            for rel in self._managed_files(self.root):
                z.write(self.root / rel, rel)
        for old in sorted(BACKUP_DIR.glob("harbor-*.zip"), key=lambda p: p.stat().st_mtime)[:-3]:
            try:
                old.unlink()
            except OSError:
                pass

    def apply(self, new_root: Path) -> None:
        self.log("Installing files ...", 65)
        new_files = self._managed_files(new_root)
        for i, rel in enumerate(new_files):
            dest = self.root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            existed = dest.exists()
            tmp = dest.with_name(dest.name + ".harbor-new")
            shutil.copyfile(new_root / rel, tmp)
            os.replace(tmp, dest)
            if not existed:
                self.created.append(rel)
            if i % 25 == 0:
                self.log("Installing files ... %d/%d" % (i + 1, len(new_files)), 65 + int(15 * i / max(1, len(new_files))))
        keep = set(new_files)
        for rel in self._managed_files(self.root):            # remove modules the new release no longer has
            if rel not in keep and rel.split("/")[0] in MANAGED_DIRS:
                try:
                    (self.root / rel).unlink()
                except OSError:
                    pass

    def rollback(self) -> bool:
        if not self.backup or not self.backup.is_file():
            return False
        try:
            for rel in self.created:
                try:
                    (self.root / rel).unlink()
                except OSError:
                    pass
            with zipfile.ZipFile(self.backup) as z:
                for name in z.namelist():
                    dest = self.root / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(z.read(name))
            return True
        except (OSError, zipfile.BadZipFile):
            return False

    # -- step 4: dependencies + smoke test -------------------------------------------------------
    def install_deps(self, new_root: Path, old_req_hash: str) -> None:
        req = self.root / "requirements.txt"
        if not (self.deps and req.is_file() and _sha256(req) != old_req_hash):
            return
        self.log("Installing updated dependencies (pip) ...", 84)
        try:
            p = subprocess.run([self.python, "-m", "pip", "install", "-r", str(req), "--disable-pip-version-check", "-q"],
                               capture_output=True, text=True, timeout=600, cwd=str(self.root))
            self.log("Dependencies installed." if p.returncode == 0 else "pip reported a problem (continuing): " + (p.stderr or p.stdout)[-160:].strip(), 90)
        except (OSError, subprocess.TimeoutExpired) as e:
            self.log("Could not run pip (continuing): %s" % e, 90)

    def smoke_test(self) -> bool:
        self.log("Testing the new version ...", 93)
        try:
            env = dict(os.environ, PYTHONPATH=str(self.root), HARBOR_NO_UPDATE="1")
            p = subprocess.run([self.python, str(self.root / "main.py"), "--version"], capture_output=True, text=True, timeout=60,
                               cwd=str(self.root), env=env, stdin=subprocess.DEVNULL)
            return p.returncode == 0 and self.info["version"] in p.stdout
        except (OSError, subprocess.TimeoutExpired):
            return False

    # -- orchestration ---------------------------------------------------------------------------
    def run(self) -> dict:
        if install_kind(self.root) != "folder":
            raise UpdateError("this installation (%s) is not updated in place" % install_kind(self.root))
        _acquire_lock()
        tmp = Path(tempfile.mkdtemp(prefix="harbor_update_"))
        try:
            new_root = self.download(tmp)
            self.scan(new_root)
            old_req = _sha256(self.root / "requirements.txt")
            try:
                self.make_backup()
                self.apply(new_root)
            except OSError as e:
                ok = self.rollback()
                raise UpdateError("could not write to %s (%s)%s. Run Harbor CLI from a folder you can write to." %
                                  (self.root, e, "; previous version restored" if ok else ""))
            self.install_deps(new_root, old_req)
            if not self.smoke_test():
                restored = self.rollback()
                raise UpdateError("the new version failed its self-test; %s" % ("the previous version was restored" if restored else "ROLLBACK FAILED - reinstall from %s" % self.info.get("repo", REPO)))
            self.log("Update complete.", 100)
            state = load_state()
            state.update(last_update=time.time(), updated_from=VERSION, updated_to=self.info["version"])
            state.pop("skipped", None)
            save_state(state)
            return {"status": "updated", "from": VERSION, "to": self.info["version"], "backup": str(self.backup)}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            _release_lock()


# ------------------------------------------------------------------------------------------- hand-off to the installer
def run_installer(info: dict, root: Path, mode: str, python: str | None = None) -> dict:
    """Run the installer window (a separate process) and wait for it. Returns its result dict {"status": ...}."""
    fd, jobfile = tempfile.mkstemp(prefix="harbor_job_", suffix=".json")
    os.close(fd)
    result_file = jobfile + ".result"
    try:
        Path(jobfile).write_text(json.dumps({"info": info, "root": str(root), "mode": mode, "python": python or sys.executable,
                                             "from": VERSION}), encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(root), HARBOR_NO_UPDATE="1")
        rc = subprocess.call([python or sys.executable, "-m", "harbor.updater_ui", jobfile], cwd=str(root), env=env)
        try:
            return json.loads(Path(result_file).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status = {v: k for k, v in EXIT.items()}.get(rc, "failed")
            return {"status": status, "error": "installer exited with code %s" % rc}
    finally:
        for f in (jobfile, result_file):
            try:
                os.unlink(f)
            except OSError:
                pass


def relaunch(old_version: str) -> None:
    """Start Harbor CLI again with the same arguments, in the same terminal. Does not return."""
    argv = list(getattr(sys, "orig_argv", None) or ([sys.executable, sys.argv[0]] + sys.argv[1:]))
    argv[0] = sys.executable
    argv = [a for a in argv if a not in ("--update", "--check-update")]
    env = dict(os.environ, HARBOR_JUST_UPDATED=old_version, HARBOR_NO_UPDATE_ONCE="1")
    try:
        import atexit
        atexit._run_exitfuncs()                      # stop llama-server, MCP servers, background processes before the restart
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        sys.exit(subprocess.call(argv, env=env))
    os.execve(argv[0], argv, env)


def startup_update(cfg: dict, force: bool = False, report=print) -> None:
    """Called once at start-up (interactive sessions). Returns normally unless an update was installed (then it restarts Harbor)."""
    mode = str(cfg.get("auto_update", "auto")).lower()
    if mode == "off" and not force:
        return
    info, status = check(cfg, force=force, timeout=float(cfg.get("update_timeout", 3.0)))
    if force and status != "available":
        report({"current": "Harbor CLI %s is up to date." % VERSION, "unreachable": "Could not reach GitHub to check for updates.",
                "bad-repo": "update_repo must be a github.com owner/repo.", "skipped": "Up to date (a newer version was skipped)."}.get(status, "No update available (%s)." % status))
    if not info:
        return
    kind = install_kind()
    if kind != "folder":
        hint = {"git": "this is a git checkout - run `git pull`", "pip": "installed with pip - run `pip install --upgrade git+https://github.com/%s`" % info["repo"]}.get(kind, "this installation cannot update itself")
        report("Harbor CLI %s is available (you have %s): %s." % (info["version"], VERSION, hint))
        return
    report("Harbor CLI %s is available (you have %s). Opening the installer ..." % (info["version"], VERSION))
    result = run_installer(info, install_root(), "ask" if (mode == "ask" or force and mode == "off") else "auto")
    status = result.get("status")
    if status == "updated":
        relaunch(VERSION)
    elif status == "skipped":
        skip_version(info["version"])
    elif status == "failed":
        report("Update failed: %s. Continuing with Harbor CLI %s." % (result.get("error", "unknown error"), VERSION))
