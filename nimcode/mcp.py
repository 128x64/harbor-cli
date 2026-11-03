"""Dependency-free Model Context Protocol (MCP) client: stdio, streamable-HTTP and legacy-SSE transports, plus
discovery of the MCP configuration VS Code uses (.vscode/mcp.json, user mcp.json, settings.json), Claude-style
.mcp.json / claude_desktop_config.json and Cursor's mcp.json.

VS Code format (also supported: `${workspaceFolder}`, `${userHome}`, `${env:NAME}`, `${input:id}` with an "inputs" list,
"envFile", "disabled", type "stdio" | "http" | "sse"):

    {"inputs": [{"id": "gh", "type": "promptString", "description": "GitHub token", "password": true}],
     "servers": {
        "fetch":  {"type": "stdio", "command": "uvx", "args": ["mcp-server-fetch"]},
        "github": {"type": "http", "url": "https://api.githubcopilot.com/mcp/", "headers": {"Authorization": "Bearer ${input:gh}"}}}}
"""
from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import VERSION
from .core import Tool, Registry
from .gitfinder import tool_env
from .util import IS_WIN, decode_bytes, have

PROTOCOL = "2025-06-18"


class MCPError(Exception):
    pass


# ---------------------------------------------------------------- results
def format_result(result: dict) -> str:
    parts = []
    for item in result.get("content") or []:
        t = item.get("type")
        if t == "text":
            parts.append(item.get("text", ""))
        elif t == "image":
            parts.append("[image %s, %d base64 chars - not displayed]" % (item.get("mimeType", ""), len(item.get("data", ""))))
        elif t == "resource":
            r = item.get("resource", {})
            parts.append(r.get("text") or "[resource %s]" % r.get("uri", ""))
        elif t == "resource_link":
            parts.append("[resource link %s]" % item.get("uri", ""))
        else:
            parts.append(json.dumps(item)[:500])
    if not parts and result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], indent=2))
    text = "\n".join(parts) or "(empty result)"
    return ("Error: " + text) if result.get("isError") else text


# ---------------------------------------------------------------- JSON-RPC clients
class _Base:
    name = "?"

    def __init__(self):
        self.tools: list = []
        self.server_info: dict = {}
        self.capabilities: dict = {}
        self.protocol = PROTOCOL
        self.roots: list = []
        self._ids = itertools.count(1)

    def initialize(self, timeout=30):
        res = self.request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {"roots": {"listChanged": False}},
                                          "clientInfo": {"name": "nimcode", "version": VERSION}}, timeout)
        self.server_info = res.get("serverInfo", {})
        self.capabilities = res.get("capabilities", {}) or {}
        self.protocol = res.get("protocolVersion", PROTOCOL)
        self.notify("notifications/initialized")
        return res

    def list_tools(self, timeout=30) -> list:
        tools, cursor = [], None
        for _ in range(50):
            res = self.request("tools/list", {"cursor": cursor} if cursor else {}, timeout)
            tools += res.get("tools", [])
            cursor = res.get("nextCursor")
            if not cursor:
                break
        self.tools = tools
        return tools

    def call_tool(self, name: str, arguments: dict, timeout=300) -> str:
        try:
            return format_result(self.request("tools/call", {"name": name, "arguments": arguments or {}}, timeout))
        except MCPError as e:
            return "Error: MCP server '%s': %s" % (self.name, e)

    def list_resources(self, timeout=30) -> list:
        return self.request("resources/list", {}, timeout).get("resources", [])

    def read_resource(self, uri: str, timeout=60) -> str:
        res = self.request("resources/read", {"uri": uri}, timeout)
        out = []
        for c in res.get("contents", []):
            out.append(c.get("text") if "text" in c else "[binary %s, %d base64 chars]" % (c.get("mimeType", ""), len(c.get("blob", ""))))
        return "\n".join(out) or "(empty resource)"


class _RpcClient(_Base):
    """Shared request/response bookkeeping for transports that carry out-of-band responses (stdio, SSE)."""

    def __init__(self):
        super().__init__()
        self.pending: dict = {}

    def _send(self, obj):  # pragma: no cover - transport specific
        raise NotImplementedError

    def _answer_request(self, msg):
        m = msg["method"]
        if m == "ping":
            reply = {"jsonrpc": "2.0", "id": msg["id"], "result": {}}
        elif m == "roots/list":
            reply = {"jsonrpc": "2.0", "id": msg["id"], "result": {"roots": [{"uri": Path(r).resolve().as_uri(), "name": Path(r).name} for r in self.roots]}}
        else:
            reply = {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "Method not supported"}}
        try:
            self._send(reply)
        except MCPError:
            pass

    def _dispatch(self, msg):
        if not isinstance(msg, dict):
            return
        if "method" in msg and "id" in msg:
            self._answer_request(msg)
        elif "id" in msg and msg["id"] in self.pending:
            ev, slot = self.pending.pop(msg["id"])
            slot.update(msg)
            ev.set()

    def _fail_all(self, reason):
        for ev, slot in list(self.pending.values()):
            slot["error"] = {"message": reason}
            ev.set()
        self.pending.clear()

    def request(self, method, params=None, timeout=60):
        rid = next(self._ids)
        ev, slot = threading.Event(), {}
        self.pending[rid] = (ev, slot)
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        if not ev.wait(timeout):
            self.pending.pop(rid, None)
            raise MCPError("timed out after %ss waiting for %s" % (timeout, method))
        if "error" in slot:
            err = slot["error"]
            raise MCPError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
        return slot.get("result") or {}

    def notify(self, method, params=None):
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})


_INSTALL_HINTS = {
    "npx": "it needs Node.js (https://nodejs.org). Prefer the Python version of this server if one exists.",
    "npm": "it needs Node.js (https://nodejs.org).", "node": "it needs Node.js (https://nodejs.org).",
    "docker": "it needs Docker Desktop, which is not installed.", "uvx": "install it with: pip install uv",
    "uv": "install it with: pip install uv", "pipx": "install it with: pip install pipx", "dotnet": "it needs the .NET SDK.",
    "deno": "it needs Deno.", "bunx": "it needs Bun.", "bun": "it needs Bun."}


def resolve_command(command: str, args: list):
    """Find the executable for an MCP server. Shims uvx -> `python -m uv tool run` and python -> this interpreter."""
    env_path = tool_env().get("PATH")
    found = shutil.which(command, path=env_path)
    base = Path(command).name.lower()
    for suf in (".exe", ".cmd", ".bat"):
        if base.endswith(suf):
            base = base[:-len(suf)]
    if found:
        return found, list(args)
    if base == "uvx" and have("uv"):
        return sys.executable, ["-m", "uv", "tool", "run"] + list(args)
    if base == "uv" and have("uv"):
        return sys.executable, ["-m", "uv"] + list(args)
    if base in ("python", "python3", "py"):
        a = list(args)
        if base == "py" and a and re.match(r"^-\d", a[0]):
            a = a[1:]
        return sys.executable, a
    raise MCPError("command '%s' was not found; %s" % (command, _INSTALL_HINTS.get(base, "check that it is installed and on PATH.")))


def load_env_file(path) -> dict:
    out = {}
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("\"'")
    except OSError:
        pass
    return out


class StdioMCP(_RpcClient):
    def __init__(self, name, command, args=(), env=None, cwd=None):
        super().__init__()
        self.name, self.command, self.args, self.env, self.cwd = name, command, list(args or []), env or {}, cwd
        self.wlock = threading.Lock()
        self.stderr_tail: list = []
        self.proc = None
        if cwd:
            self.roots = [cwd]

    def start(self):
        exe, args = resolve_command(self.command, self.args)
        env = tool_env()
        env.update({k: os.path.expandvars(str(v)) for k, v in self.env.items()})
        kw = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if IS_WIN else {"start_new_session": True}
        try:
            self.proc = subprocess.Popen([exe] + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, env=env, cwd=self.cwd, **kw)
        except OSError as e:
            raise MCPError("cannot start '%s': %s" % (self.command, e))
        threading.Thread(target=self._read_loop, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _send(self, obj):
        data = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
        with self.wlock:
            try:
                self.proc.stdin.write(data)
                self.proc.stdin.flush()
            except (OSError, ValueError) as e:
                raise MCPError("server connection lost: %s" % e)

    def _drain_stderr(self):
        try:
            for line in iter(self.proc.stderr.readline, b""):
                self.stderr_tail = (self.stderr_tail + [decode_bytes(line).rstrip()])[-20:]
        except Exception:
            pass

    def _read_loop(self):
        try:
            for line in iter(self.proc.stdout.readline, b""):
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue  # noise on stdout
                self._dispatch(msg)
        finally:
            self._fail_all("server exited. stderr: " + " | ".join(self.stderr_tail[-3:]))

    def close(self):
        if self.proc and self.proc.poll() is None:
            for fn in (lambda: self.proc.stdin.close(), lambda: self.proc.terminate(), lambda: self.proc.wait(timeout=3)):
                try:
                    fn()
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass


class SseMCP(_RpcClient):
    """Legacy HTTP+SSE transport: GET opens an event stream that announces a POST endpoint."""

    def __init__(self, name, url, headers=None):
        super().__init__()
        self.name, self.url, self.headers = name, url, dict(headers or {})
        self.endpoint, self.ready, self.resp, self.error, self.closed = None, threading.Event(), None, None, False

    def start(self, timeout=20):
        threading.Thread(target=self._stream, daemon=True).start()
        if not self.ready.wait(timeout) or not self.endpoint:
            raise MCPError("SSE server did not announce an endpoint (%s)" % (self.error or "timeout"))

    def _headers(self, extra=None):
        h = {"User-Agent": "nimcode/" + VERSION}
        h.update(self.headers)
        h.update(extra or {})
        return h

    def _stream(self):
        try:
            req = urllib.request.Request(self.url, headers=self._headers({"Accept": "text/event-stream"}))
            self.resp = urllib.request.urlopen(req, timeout=600)
            event, data = "message", []
            for raw in self.resp:
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line == "":
                    if data:
                        self._on_event(event, "\n".join(data))
                    event, data = "message", []
                elif line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip())
        except Exception as e:
            self.error = e
        finally:
            self._fail_all("SSE stream closed")
            self.ready.set()

    def _on_event(self, event, data):
        if event == "endpoint":
            self.endpoint = urllib.parse.urljoin(self.url, data.strip())
            self.ready.set()
            return
        try:
            self._dispatch(json.loads(data))
        except ValueError:
            pass

    def _send(self, obj):
        if not self.endpoint:
            raise MCPError("no SSE endpoint")
        req = urllib.request.Request(self.endpoint, data=json.dumps(obj).encode(), headers=self._headers({"Content-Type": "application/json"}), method="POST")
        try:
            urllib.request.urlopen(req, timeout=30).close()
        except urllib.error.HTTPError as e:
            raise MCPError("HTTP %d: %s" % (e.code, decode_bytes(e.read())[:200]))
        except urllib.error.URLError as e:
            raise MCPError("connection failed: %s" % e)

    def close(self):
        """Never block: a reader thread holds the response's buffer lock, so shut the socket down first, then close."""
        self.closed = True
        resp = self.resp
        if resp is None:
            return
        try:
            sock = resp.fp.raw._sock
            sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        threading.Thread(target=lambda: resp.close(), daemon=True).start()


class HttpMCP(_Base):
    """Streamable HTTP transport (POST per request; JSON or SSE response)."""

    def __init__(self, name, url, headers=None):
        super().__init__()
        self.name, self.url, self.headers = name, url, {k: os.path.expandvars(str(v)) for k, v in (headers or {}).items()}
        self.session_id = None

    def start(self):
        pass

    def _post(self, body, timeout):
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", "User-Agent": "nimcode/" + VERSION}
        h.update(self.headers)
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        if body.get("method") != "initialize":
            h["MCP-Protocol-Version"] = self.protocol
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=h, method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            raise MCPError("HTTP %d: %s" % (e.code, decode_bytes(e.read())[:300]))
        except urllib.error.URLError as e:
            raise MCPError("connection failed: %s" % e)
        sid = resp.headers.get("Mcp-Session-Id")
        if sid:
            self.session_id = sid
        return resp

    def request(self, method, params=None, timeout=60):
        rid = next(self._ids)
        resp = self._post({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}, timeout)
        try:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            msgs = []
            if "text/event-stream" in ctype:
                data = []
                for raw in resp:
                    line = raw.decode("utf-8", "replace").rstrip("\r\n")
                    if line.startswith("data:"):
                        data.append(line[5:].lstrip())
                    elif line == "" and data:
                        try:
                            msgs.append(json.loads("\n".join(data)))
                        except ValueError:
                            pass
                        data = []
                        if any(isinstance(m, dict) and m.get("id") == rid for m in msgs):
                            break
                if data:
                    try:
                        msgs.append(json.loads("\n".join(data)))
                    except ValueError:
                        pass
            else:
                body = json.loads(decode_bytes(resp.read()) or "null")
                msgs = body if isinstance(body, list) else [body]
        finally:
            resp.close()
        for m in msgs:
            if isinstance(m, dict) and m.get("id") == rid:
                if "error" in m:
                    raise MCPError(m["error"].get("message", str(m["error"])))
                return m.get("result") or {}
        raise MCPError("no response to %s" % method)

    def notify(self, method, params=None):
        try:
            self._post({"jsonrpc": "2.0", "method": method, "params": params or {}}, 15).close()
        except MCPError:
            pass

    def close(self):
        if self.session_id:
            try:
                req = urllib.request.Request(self.url, headers=dict(self.headers, **{"Mcp-Session-Id": self.session_id}), method="DELETE")
                urllib.request.urlopen(req, timeout=5).close()
            except Exception:
                pass


# ---------------------------------------------------------------- config discovery (VS Code, Claude, Cursor, NimCode)
def strip_jsonc(text: str) -> str:
    """Remove // and /* */ comments and trailing commas (VS Code files are JSONC)."""
    out, i, n = [], 0, len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 1
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
            continue
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        elif ch == ",":
            j = i + 1
            while j < n:
                if text[j].isspace():
                    j += 1
                elif text.startswith("//", j):
                    while j < n and text[j] != "\n":
                        j += 1
                elif text.startswith("/*", j):
                    k = text.find("*/", j + 2)
                    j = n if k < 0 else k + 2
                else:
                    break
            if j < n and text[j] in "}]":
                i += 1
                continue
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def loads_jsonc(text: str):
    return json.loads(strip_jsonc(text.lstrip("\ufeff")))


@dataclass
class Source:
    path: Path
    label: str
    servers: dict
    inputs: list = field(default_factory=list)
    project: bool = False


def _candidate_files(cwd: Path, include_project: bool):
    home = Path.home()
    appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
    code_dirs = []
    if IS_WIN:
        code_dirs = [appdata / "Code" / "User", appdata / "Code - Insiders" / "User", appdata / "VSCodium" / "User"]
    elif sys.platform == "darwin":
        base = home / "Library" / "Application Support"
        code_dirs = [base / "Code" / "User", base / "Code - Insiders" / "User", base / "VSCodium" / "User"]
    else:
        code_dirs = [home / ".config" / "Code" / "User", home / ".config" / "Code - Insiders" / "User", home / ".config" / "VSCodium" / "User"]
    out = []
    if IS_WIN:
        out.append((appdata / "Claude" / "claude_desktop_config.json", "Claude Desktop", False))
    elif sys.platform == "darwin":
        out.append((home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json", "Claude Desktop", False))
    out.append((home / ".cursor" / "mcp.json", "Cursor", False))
    for d in code_dirs:
        out.append((d / "settings.json", "VS Code settings", False))
        out.append((d / "mcp.json", "VS Code user", False))
        if (d / "profiles").is_dir():
            for prof in sorted((d / "profiles").glob("*/mcp.json")):
                out.append((prof, "VS Code profile", False))
    out.append((home / ".nimcode" / "mcp.json", "NimCode", False))
    if include_project:
        for rel, label in ((".cursor/mcp.json", "Cursor project"), (".vscode/mcp.json", "VS Code workspace"),
                           (".mcp.json", "project .mcp.json"), (".nimcode/mcp.json", "NimCode project")):
            out.append((cwd / rel, label, True))
    return out


def _servers_from(data) -> dict:
    if not isinstance(data, dict):
        return {}
    for key in ("servers", "mcpServers"):
        if isinstance(data.get(key), dict):
            return data[key]
    mcp = data.get("mcp")  # VS Code settings.json: {"mcp": {"servers": {...}}}
    if isinstance(mcp, dict) and isinstance(mcp.get("servers"), dict):
        return mcp["servers"]
    return {}


def read_source(path: Path, label: str = "", project: bool = False):
    try:
        data = loads_jsonc(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    servers = {k: v for k, v in _servers_from(data).items() if isinstance(v, dict)}
    inputs = list(data.get("inputs") or []) + list((data.get("mcp") or {}).get("inputs") or []) if isinstance(data, dict) else []
    return Source(Path(path), label or str(path), servers, [i for i in inputs if isinstance(i, dict)], project) if servers else None


def discover_sources(cwd: Path, include_project: bool, import_vscode: bool = True, extra=()) -> list:
    """All MCP config sources, lowest precedence first."""
    sources = []
    for path, label, project in _candidate_files(Path(cwd), include_project):
        if not import_vscode and label.startswith(("VS Code", "Cursor", "Claude")):
            continue
        if path.is_file():
            s = read_source(path, label, project)
            if s:
                sources.append(s)
    for p in extra:
        s = read_source(Path(p), "--mcp-config")
        if s:
            sources.append(s)
    return sources


_VAR = re.compile(r"\$\{([^}]+)\}")


class MissingInput(Exception):
    pass


def _expand(value, cwd: Path, resolve_input):
    def sub(s: str) -> str:
        def rep(m):
            k = m.group(1)
            if k in ("workspaceFolder", "workspaceRoot", "cwd"):
                return str(cwd)
            if k == "workspaceFolderBasename":
                return cwd.name
            if k == "userHome":
                return str(Path.home())
            if k in ("pathSeparator", "/"):
                return os.sep
            if k.startswith("env:"):
                return os.environ.get(k[4:], "")
            if k.startswith("input:"):
                return resolve_input(k[6:])
            return os.environ.get(k, "")
        return _VAR.sub(rep, s)

    if isinstance(value, str):
        return sub(value)
    if isinstance(value, list):
        return [_expand(v, cwd, resolve_input) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v, cwd, resolve_input) for k, v in value.items()}
    return value


def resolve_specs(sources: list, cwd: Path, ask_input=None, report=print) -> dict:
    """Merge sources (later wins), expand variables and prompt for ${input:...} values. Returns {name: normalized spec}."""
    cwd = Path(cwd)
    merged: dict = {}
    input_defs: dict = {}
    for s in sources:
        for i in s.inputs:
            if i.get("id"):
                input_defs[i["id"]] = i
        for name, raw in s.servers.items():
            merged[name] = (raw, s)
    cache: dict = {}

    def resolve_input(iid):
        if iid not in cache:
            d = input_defs.get(iid, {"id": iid})
            val = os.environ.get("NIMCODE_INPUT_" + re.sub(r"\W", "_", iid).upper())
            if val is None and ask_input:
                val = ask_input(iid, d)
            if val is None and d.get("default") is not None:
                val = str(d["default"])
            cache[iid] = val
        if cache[iid] is None:
            raise MissingInput(iid)
        return cache[iid]

    specs = {}
    for name, (raw, src) in merged.items():
        if raw.get("disabled"):
            continue
        try:
            r = _expand(raw, cwd, resolve_input)
        except MissingInput as e:
            report("MCP '%s' skipped: no value for input '%s' (set NIMCODE_INPUT_%s)" % (name, e, str(e).upper()))
            continue
        typ = (r.get("type") or ("http" if r.get("url") else "stdio")).lower()
        typ = {"streamable-http": "http", "streamablehttp": "http"}.get(typ, typ)
        env = {}
        if r.get("envFile"):
            ef = Path(str(r["envFile"]))
            env.update(load_env_file(ef if ef.is_absolute() else cwd / ef))
        env.update({k: str(v) for k, v in (r.get("env") or {}).items()})
        specs[name] = {"type": typ, "explicit_type": bool(r.get("type")), "command": r.get("command"), "args": r.get("args") or [],
                       "env": env, "url": r.get("url"), "headers": r.get("headers") or {}, "cwd": r.get("cwd"),
                       "timeout": r.get("timeout", 45), "source": src.label}
    return specs


def read_servers(path) -> dict:
    """Backward-compatible helper: {name: spec} from one config file (no prompting)."""
    s = read_source(Path(path))
    if not s:
        return {}
    return resolve_specs([s], Path.cwd(), None, lambda m: None)


def make_client(name: str, spec: dict, cwd):
    typ = spec.get("type", "stdio")
    if typ in ("http", "sse"):
        if not spec.get("url"):
            raise MCPError("server '%s' needs a url" % name)
        return SseMCP(name, spec["url"], spec.get("headers")) if typ == "sse" else HttpMCP(name, spec["url"], spec.get("headers"))
    if not spec.get("command"):
        raise MCPError("server '%s' needs a command or url" % name)
    return StdioMCP(name, spec["command"], spec.get("args"), spec.get("env"), spec.get("cwd") or str(cwd))


def start_client(name: str, spec: dict, cwd):
    """Create, start and initialize a client (HTTP falls back to legacy SSE when not explicitly typed). Tools are listed."""
    c = make_client(name, spec, cwd)
    c.roots = [str(cwd)]
    try:
        c.start()
        c.initialize(spec.get("timeout", 45))
    except MCPError:
        if spec.get("type") == "http" and not spec.get("explicit_type"):
            try:
                c.close()
            except Exception:
                pass
            c = SseMCP(name, spec["url"], spec.get("headers"))
            c.roots = [str(cwd)]
            c.start()
            c.initialize(spec.get("timeout", 45))
        else:
            raise
    c.list_tools()
    return c


def connect_servers(specs: dict, cwd, report=print) -> list:
    results: dict = {}

    def work(name, spec):
        try:
            results[name] = start_client(name, spec, cwd)
        except Exception as e:  # noqa: BLE001
            results[name] = e

    threads = [threading.Thread(target=work, args=(n, s), daemon=True) for n, s in specs.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join(90)
    clients = []
    for name in specs:
        r = results.get(name)
        if isinstance(r, _Base):
            clients.append(r)
            report("MCP '%s' (%s): %d tools" % (name, specs[name].get("source", ""), len(r.tools)))
        else:
            report("MCP '%s' failed: %s" % (name, r or "timeout"))
    return clients


def register_mcp_tools(reg: Registry, clients: list) -> int:
    n = 0
    for c in clients:
        safe_server = re.sub(r"[^a-zA-Z0-9_-]", "_", c.name)
        for t in c.tools:
            base = "mcp__%s__%s" % (safe_server, re.sub(r"[^a-zA-Z0-9_-]", "_", t.get("name", "tool")))
            name, i = base[:64], 2
            while name in reg.tools:
                name, i = base[:60] + "_%d" % i, i + 1
            schema = dict(t.get("inputSchema") or {})
            schema["type"] = "object"
            schema.setdefault("properties", {})
            ro = bool((t.get("annotations") or {}).get("readOnlyHint"))
            desc = (t.get("description") or t.get("title") or t.get("name", ""))[:1000] + " [MCP server: %s]" % c.name

            def make(client, tool_name):
                return lambda ctx, **kw: client.call_tool(tool_name, kw)
            reg.add(Tool(name=name, description=desc, schema=schema, func=make(c, t["name"]),
                         group="mcp:" + c.name, ask=not ro, read_only=ro, filter_args=False))
            n += 1
    res_clients = {c.name: c for c in clients if "resources" in (c.capabilities or {})}
    if res_clients:
        def pick(server):
            if server and server in res_clients:
                return res_clients[server]
            return next(iter(res_clients.values())) if len(res_clients) == 1 and not server else None

        def list_resources(ctx, server=""):
            out = []
            for name, c in ([(server, res_clients[server])] if server in res_clients else list(res_clients.items())):
                try:
                    for r in c.list_resources():
                        out.append("%s | %s | %s" % (name, r.get("uri"), r.get("name", "")))
                except MCPError as e:
                    out.append("%s | error: %s" % (name, e))
            return "\n".join(out) or "No resources."

        def read_resource(ctx, uri, server=""):
            c = pick(server)
            if c is None:
                return "Error: specify server (one of: %s)." % ", ".join(res_clients)
            try:
                return c.read_resource(uri)
            except MCPError as e:
                return "Error: %s" % e
        reg.add(Tool("mcp_list_resources", "List resources exposed by MCP servers (documents, files, database rows...).",
                     {"type": "object", "properties": {"server": {"type": "string"}}, "required": []}, list_resources,
                     group="mcp", read_only=True))
        reg.add(Tool("mcp_read_resource", "Read an MCP resource by URI.",
                     {"type": "object", "properties": {"uri": {"type": "string"}, "server": {"type": "string"}}, "required": ["uri"]},
                     read_resource, group="mcp", read_only=True, summary_key="uri"))
    return n
