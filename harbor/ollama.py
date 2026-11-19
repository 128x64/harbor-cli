"""Ollama backend (native /api/chat): streaming, native tool calling, thinking models, per-request context size.

Why native instead of Ollama's OpenAI-compatible endpoint: only /api/chat lets us set `num_ctx` per request. Ollama
silently truncates prompts to its (small) default context otherwise, which ruins agentic work with big prompts.
Server: http://localhost:11434 by default (OLLAMA_HOST is honoured). Ollama's cloud API (https://ollama.com) works with
OLLAMA_API_KEY.
"""
from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
import uuid

from . import VERSION
from .client import NIMError

DEFAULT_HOST = "http://localhost:11434"


def normalize_host(host: str = "") -> str:
    """OLLAMA_HOST style values ('127.0.0.1:11434', '0.0.0.0', 'http://x:1/v1') -> 'http://host:port'."""
    h = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).strip().rstrip("/")
    if not re.match(r"^https?://", h):
        h = ("https://" if "ollama.com" in h else "http://") + h
    h = re.sub(r"/(v1|api)$", "", h)
    h = h.replace("://0.0.0.0", "://127.0.0.1")
    if not re.search(r":\d+$", h) and h.startswith("http://") and "ollama.com" not in h:
        h += ":11434"
    return h


def to_ollama_messages(messages: list) -> list:
    """OpenAI-style history (tool_calls with JSON-string arguments, role=tool) -> Ollama's /api/chat format."""
    names = {}
    out = []
    for m in messages:
        role = m["role"]
        content = m.get("content") or ""
        images = []
        if isinstance(content, list):          # OpenAI-style multimodal parts (analyze_image)
            texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            images = [p["image_url"]["url"].split(",", 1)[1] for p in content
                      if isinstance(p, dict) and p.get("type") == "image_url" and str(p.get("image_url", {}).get("url", "")).startswith("data:")]
            content = "\n".join(texts)
        elif not isinstance(content, str):
            content = json.dumps(content)
        if role == "user" and images:
            out.append({"role": "user", "content": content, "images": images})
            continue
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                names[tc["id"]] = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except ValueError:
                    args = {}
                calls.append({"function": {"name": tc["function"]["name"], "arguments": args}})
            out.append({"role": "assistant", "content": content, "tool_calls": calls})
        elif role == "tool":
            out.append({"role": "tool", "content": content, "tool_name": names.get(m.get("tool_call_id"), "")})
        else:
            out.append({"role": role, "content": content})
    return out


class OllamaClient:
    def __init__(self, host: str = "", num_ctx: int = 16384, keep_alive: str = "30m", timeout: int = 900, api_key: str = ""):
        self.host = normalize_host(host)
        self.base_url, self.num_ctx, self.keep_alive, self.timeout = self.host, num_ctx, keep_alive, timeout
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY") or "ollama"
        self.on_retry = None
        self.on_chars = None
        self.backend_desc = "Ollama (%s)" % self.host

    # -- plumbing ----------------------------------------------------------------
    @staticmethod
    def _err(body: str) -> str:
        try:
            o = json.loads(body)
            if isinstance(o, dict) and o.get("error"):
                return str(o["error"])
        except ValueError:
            pass
        return body.strip()[:400] or "empty error body"

    def _open(self, path, payload=None):
        headers = {"User-Agent": "harbor/" + VERSION, "Accept": "application/json"}
        if self.api_key != "ollama":
            headers["Authorization"] = "Bearer " + self.api_key
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        delay = 1
        for attempt in range(4):
            req = urllib.request.Request(self.host + path, data=data, headers=headers, method="POST" if data is not None else "GET")
            try:
                return urllib.request.urlopen(req, timeout=self.timeout)
            except urllib.error.HTTPError as e:
                body = self._err(e.read().decode("utf-8", "replace"))
                if e.code in (429, 502, 503, 504) and attempt < 3:
                    if self.on_retry:
                        self.on_retry("Ollama busy (HTTP %d); retrying in %ds..." % (e.code, delay))
                    time.sleep(delay)
                    delay *= 2
                    continue
                if e.code == 404 and "not found" in body.lower():
                    body += " - download it first with: ollama pull <model>"
                raise NIMError(e.code, body) from None
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
                if attempt < 1:
                    time.sleep(1)
                    continue
                raise NIMError(0, "Cannot reach Ollama at %s (%s). Start it with `ollama serve` or open the Ollama app, "
                                  "or set OLLAMA_HOST / --ollama-url." % (self.host, getattr(e, "reason", e))) from None
        raise NIMError(0, "Exhausted retries")

    def _tick(self, s) -> None:
        if self.on_chars and s:
            self.on_chars(len(s))

    # -- public ------------------------------------------------------------------
    def is_up(self, timeout=1.5) -> bool:
        try:
            with urllib.request.urlopen(self.host + "/api/version", timeout=timeout) as r:
                return r.status == 200
        except Exception:
            return False

    def preload(self, model: str) -> None:
        """Load the model into RAM/VRAM now (Ollama: an empty prompt loads it). Uses the same num_ctx as chat requests,
        otherwise Ollama would unload and reload the model on the first message."""
        payload = {"model": model, "prompt": "", "stream": False, "keep_alive": self.keep_alive, "options": {"num_ctx": self.num_ctx}}
        with self._open("/api/generate", payload) as r:
            r.read()

    def list_models(self):
        with self._open("/api/tags") as r:
            return sorted(m.get("name", "") for m in json.loads(r.read().decode("utf-8", "replace")).get("models", []) if m.get("name"))

    def chat_stream(self, model, messages, tools=None, max_tokens=4096, temperature=0.3, on_text=None, on_reasoning=None):
        payload = {"model": model, "messages": to_ollama_messages(messages), "stream": True, "keep_alive": self.keep_alive,
                   "options": {"num_ctx": self.num_ctx, "temperature": temperature, "num_predict": max_tokens}}
        if tools:
            payload["tools"] = tools
        content, reasoning, calls, done = [], [], [], {}
        resp = self._open("/api/chat", payload)
        try:
            for raw in resp:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if obj.get("error"):
                    raise NIMError(500, str(obj["error"]))
                msg = obj.get("message") or {}
                if msg.get("thinking"):
                    self._tick(msg["thinking"])
                    reasoning.append(msg["thinking"])
                    if on_reasoning:
                        on_reasoning(msg["thinking"])
                if msg.get("content"):
                    self._tick(msg["content"])
                    content.append(msg["content"])
                    if on_text:
                        on_text(msg["content"])
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    args = fn.get("arguments", {})
                    self._tick(fn.get("name", "")), self._tick(args if isinstance(args, str) else json.dumps(args))
                    calls.append({"id": tc.get("id") or "call_" + uuid.uuid4().hex[:9], "name": fn.get("name", ""),
                                  "args": args if isinstance(args, str) else json.dumps(args)})
                if obj.get("done"):
                    done = obj
        except (socket.timeout, TimeoutError, ConnectionError, urllib.error.URLError) as e:
            raise NIMError(0, "Ollama stream interrupted: %s" % e) from None
        finally:
            resp.close()
        calls = [c for c in calls if c["name"]]
        usage = None
        if done.get("prompt_eval_count") is not None:
            usage = {"prompt_tokens": done.get("prompt_eval_count", 0), "completion_tokens": done.get("eval_count", 0)}
        return {"content": "".join(content), "reasoning": "".join(reasoning), "tool_calls": calls,
                "finish_reason": "tool_calls" if calls else done.get("done_reason", "stop"), "usage": usage}
