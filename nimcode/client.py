"""NVIDIA NIM client: OpenAI-compatible chat completions with SSE streaming, retry and backoff.

Hosted endpoint: https://integrate.api.nvidia.com/v1  (Authorization: Bearer nvapi-...)
Self-hosted NIM containers expose the same API, e.g. http://localhost:8000/v1
"""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
import uuid

from . import VERSION

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_VISION_MODEL = "meta/llama-3.2-90b-vision-instruct"


class NIMError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class NIMClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, timeout: int = 300):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.on_retry = None  # callable(str)

    # -- low level ---------------------------------------------------------- #
    @staticmethod
    def _err_msg(body: str) -> str:
        try:
            obj = json.loads(body)
            if isinstance(obj, dict):
                err = obj.get("error")
                if isinstance(err, dict):
                    return str(err.get("message") or err)
                for key in ("detail", "message", "error", "title"):
                    if obj.get(key):
                        return str(obj[key])
        except Exception:
            pass
        return body.strip()[:500] or "empty error body"

    def _notify(self, msg: str) -> None:
        if self.on_retry:
            self.on_retry(msg)

    def _open(self, path: str, payload=None, stream: bool = False):
        url = self.base_url + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Accept": "text/event-stream" if stream else "application/json",
            "User-Agent": "nimcode/" + VERSION,
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        delay = 2
        for attempt in range(6):
            req = urllib.request.Request(url, data=data, headers=headers,
                                         method="POST" if data is not None else "GET")
            try:
                return urllib.request.urlopen(req, timeout=self.timeout)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")
                if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                    wait = delay
                    ra = e.headers.get("Retry-After") if e.headers else None
                    if ra and ra.strip().isdigit():
                        wait = min(int(ra), 60)
                    self._notify("HTTP %d from NIM (%s); retrying in %ds..." % (e.code, self._err_msg(body)[:80], wait))
                    time.sleep(wait)
                    delay = min(delay * 2, 30)
                    continue
                raise NIMError(e.code, self._err_msg(body)) from None
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
                if attempt < 3:
                    self._notify("Network error (%s); retrying in %ds..." % (e, delay))
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                raise NIMError(0, "Network error: %s" % e) from None
        raise NIMError(0, "Exhausted retries")

    # -- public API --------------------------------------------------------- #
    def list_models(self):
        with self._open("/models") as resp:
            obj = json.loads(resp.read().decode("utf-8", "replace"))
        return sorted(m.get("id", "") for m in obj.get("data", []) if m.get("id"))

    def chat_stream(self, model, messages, tools=None, max_tokens=8192, temperature=0.3,
                    on_text=None, on_reasoning=None):
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
        content, reasoning = [], []
        calls = {}
        finish, usage = None, None

        def handle_choice_delta(delta):
            rc = delta.get("reasoning_content") or delta.get("reasoning")
            if rc:
                reasoning.append(rc)
                if on_reasoning:
                    on_reasoning(rc)
            txt = delta.get("content")
            if txt:
                content.append(txt)
                if on_text:
                    on_text(txt)
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", len(calls))
                slot = calls.setdefault(idx, {"id": None, "name": "", "args": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name") and not slot["name"]:
                    slot["name"] = fn["name"]
                a = fn.get("arguments")
                if isinstance(a, dict):
                    slot["args"] += json.dumps(a)
                elif a:
                    slot["args"] += a

        resp = self._open("/chat/completions", payload, stream=True)
        try:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "text/event-stream" not in ctype:  # server ignored stream=true
                obj = json.loads(resp.read().decode("utf-8", "replace"))
                if obj.get("error"):
                    raise NIMError(0, self._err_msg(json.dumps(obj)))
                ch = (obj.get("choices") or [{}])[0]
                handle_choice_delta(ch.get("message") or {})
                finish, usage = ch.get("finish_reason"), obj.get("usage")
            else:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except ValueError:
                        continue
                    if obj.get("error"):
                        raise NIMError(0, self._err_msg(json.dumps(obj)))
                    if obj.get("usage"):
                        usage = obj["usage"]
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    ch = choices[0]
                    handle_choice_delta(ch.get("delta") or {})
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        except (socket.timeout, TimeoutError, ConnectionError, urllib.error.URLError) as e:
            raise NIMError(0, "Stream interrupted: %s" % e) from None
        finally:
            resp.close()

        tool_calls = []
        for idx in sorted(calls):
            s = calls[idx]
            if not s["name"]:
                continue
            tool_calls.append({
                "id": s["id"] or ("call_" + uuid.uuid4().hex[:9]),
                "name": s["name"],
                "args": s["args"] or "{}",
            })
        return {
            "content": "".join(content),
            "reasoning": "".join(reasoning),
            "tool_calls": tool_calls,
            "finish_reason": finish,
            "usage": usage,
        }
