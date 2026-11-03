"""Local model backends. If a model.gguf sits next to the main script, NimCode runs it instead of calling NVIDIA NIM.

Two engines (auto-picked in this order):
  1. llama-server  - the prebuilt llama.cpp binary (llama-server.exe on Windows; no compiler or Python wheels needed).
     Place it next to the script (or in bin/ / llama.cpp/ or on PATH). NimCode starts it on a free local port with
     --jinja and talks to its OpenAI-compatible API (native tool calling, with automatic fallback to prompt-based).
  2. llama-cpp-python - `pip install llama-cpp-python` runs the model in-process (prompt-based tool calling).
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .client import NIMClient, NIMError
from .util import IS_WIN, decode_bytes

LOG_FILE = Path.home() / ".nimcode" / "llama-server.log"


def find_local_model(dirs, explicit: str = ""):
    """model.gguf next to the main script (or an explicit path). If no model.gguf, a single *.gguf file is used."""
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    for d in dirs:
        if not d:
            continue
        d = Path(d)
        if (d / "model.gguf").is_file():
            return d / "model.gguf"
    for d in dirs:
        if d and Path(d).is_dir():
            ggufs = sorted(Path(d).glob("*.gguf"))
            if len(ggufs) == 1:
                return ggufs[0]
    return None


def find_llama_server(dirs):
    names = ["llama-server.exe", "llama-server"] if IS_WIN else ["llama-server"]
    for d in dirs:
        if not d:
            continue
        for sub in ("", "bin", "llama.cpp", "llama.cpp/bin", "llama-server"):
            for n in names:
                p = Path(d) / sub / n
                if p.is_file():
                    return str(p)
    for n in names:
        w = shutil.which(n)
        if w:
            return w
    return None


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class LlamaServer:
    """Starts llama-server for a GGUF file and exposes an OpenAI-compatible NIMClient pointed at it."""

    def __init__(self, exe, model, n_ctx=8192, gpu_layers=0, threads=0, extra=None, report=print):
        self.exe, self.model, self.n_ctx, self.gpu_layers, self.threads = exe, Path(model), n_ctx, gpu_layers, threads
        self.extra, self.report, self.proc, self.port = list(extra or []), report, None, 0

    def start(self, timeout=600) -> NIMClient:
        self.port = _free_port()
        argv = [self.exe, "-m", str(self.model), "--host", "127.0.0.1", "--port", str(self.port), "-c", str(self.n_ctx), "--jinja"]
        if self.gpu_layers:
            argv += ["-ngl", str(self.gpu_layers)]
        if self.threads:
            argv += ["-t", str(self.threads)]
        argv += self.extra
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        log = open(LOG_FILE, "wb")
        kw = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if IS_WIN else {"start_new_session": True}
        try:
            self.proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **kw)
        except OSError as e:
            raise NIMError(0, "could not start llama-server (%s): %s" % (self.exe, e))
        atexit.register(self.stop)
        base = "http://127.0.0.1:%d" % self.port
        t0, last = time.time(), 0
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                tail = decode_bytes(LOG_FILE.read_bytes()[-1500:])
                raise NIMError(0, "llama-server exited with code %s while loading the model. Log tail:\n%s" % (self.proc.returncode, tail))
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except urllib.error.HTTPError:
                pass  # 503 while the model is loading
            except (urllib.error.URLError, OSError):
                pass
            if time.time() - last > 5:
                self.report("Loading %s ... (%ds)" % (self.model.name, time.time() - t0))
                last = time.time()
            time.sleep(0.5)
        else:
            self.stop()
            raise NIMError(0, "timed out waiting for llama-server to load the model (see %s)" % LOG_FILE)
        return NIMClient("local", base + "/v1", timeout=900)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                if IS_WIN:
                    subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True, timeout=15)
                else:
                    self.proc.terminate()
                    self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass


class LlamaCppPython:
    """In-process inference with llama-cpp-python (no tools natively: wrap in TextToolClient)."""

    def __init__(self, model, n_ctx=8192, gpu_layers=0, threads=0):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise NIMError(0, "llama-cpp-python is not installed. Either put llama-server(.exe) from llama.cpp's releases next to "
                              "the script, or run: pip install llama-cpp-python "
                              "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu")
        kw = dict(model_path=str(model), n_ctx=n_ctx, n_gpu_layers=gpu_layers, verbose=False)
        if threads:
            kw["n_threads"] = threads
        self.llm = Llama(**kw)
        self._lock = __import__("threading").Lock()   # llama.cpp contexts are not thread-safe
        self.model_name = Path(model).name
        self.base_url, self.api_key, self.on_retry = "local (llama-cpp-python)", "local", None

    def list_models(self):
        return [self.model_name]

    def chat_stream(self, model, messages, tools=None, max_tokens=4096, temperature=0.3, on_text=None, on_reasoning=None):
        content, finish = [], None
        with self._lock:
            for chunk in self.llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature, stream=True):
                ch = (chunk.get("choices") or [{}])[0]
                d = ch.get("delta") or {}
                if d.get("content"):
                    content.append(d["content"])
                    if on_text:
                        on_text(d["content"])
                finish = ch.get("finish_reason") or finish
        return {"content": "".join(content), "reasoning": "", "tool_calls": [], "finish_reason": finish, "usage": None}


def make_local_client(model, engine, dirs, n_ctx, gpu_layers, threads, tool_mode, report):
    """Return (client, description). `tool_mode`: auto|native|text."""
    from .textproto import AutoToolClient, TextToolClient
    exe = find_llama_server(dirs) if engine in ("auto", "server") else None
    if engine == "server" and not exe:
        raise NIMError(0, "llama-server was not found next to the script, in bin/, or on PATH.")
    if exe:
        srv = LlamaServer(exe, model, n_ctx, gpu_layers, threads, report=report)
        inner = srv.start()
        client = TextToolClient(inner) if tool_mode == "text" else inner if tool_mode == "native" else AutoToolClient(inner, report)
        return client, "llama-server (%s)" % Path(exe).name
    inner = LlamaCppPython(model, n_ctx, gpu_layers, threads)
    return TextToolClient(inner), "llama-cpp-python"
