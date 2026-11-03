"""Prompt-based tool calling for models without native function calling (local GGUF models, odd NIM models).

Tools are described in the system prompt in the Hermes/Qwen format and the model answers with
    <tool_call>{"name": ..., "arguments": {...}}</tool_call>
which is parsed out of the token stream. Also understands Qwen3-Coder's <function=...><parameter=...> blocks, Llama-style
{"name":..., "parameters":...} objects, <think> reasoning blocks, and unterminated tool-call blocks."""
from __future__ import annotations

import json
import re
import uuid

OPEN_TAGS = {"<think>": "think", "<tool_call>": "tool"}
CLOSE_TAGS = {"think": "</think>", "tool": "</tool_call>"}


def _short(desc: str, n: int) -> str:
    d = " ".join((desc or "").split())
    cut = d.find(". ")
    if 0 < cut < n:
        d = d[:cut + 1]
    return d[:n]


def compact_spec(spec: dict, desc_len: int = 140) -> dict:
    f = spec["function"]
    params = f.get("parameters", {})
    props = {}
    for k, v in (params.get("properties") or {}).items():
        item = {"type": v.get("type", "string")}
        if v.get("enum"):
            item["enum"] = v["enum"]
        if v.get("description"):
            item["description"] = _short(v["description"], 60)
        props[k] = item
    return {"name": f["name"], "description": _short(f.get("description", ""), desc_len),
            "parameters": {"type": "object", "properties": props, "required": params.get("required", [])}}


def tools_prompt(specs: list, compact: bool = True) -> str:
    lines = [json.dumps(compact_spec(s) if compact else s["function"], separators=(",", ":")) for s in specs]
    return (
        "# Tools\n\nYou can call functions to act on the user's computer. The function signatures are inside <tools></tools>:\n"
        "<tools>\n" + "\n".join(lines) + "\n</tools>\n\n"
        "To call a function, reply with a JSON object inside <tool_call></tool_call> tags:\n"
        "<tool_call>\n{\"name\": \"<function-name>\", \"arguments\": {<args-json-object>}}\n</tool_call>\n"
        "You may write a short sentence before a call and may make several calls in one reply (one block each). "
        "After a call, STOP and wait: the result arrives in a <tool_response> message. When no more calls are needed, "
        "answer normally without any <tool_call> block.")


def convert_messages(messages: list, specs: list | None, compact: bool = True) -> list:
    """OpenAI-style messages (with tool_calls / tool roles) -> plain system/user/assistant messages."""
    first = messages[0]
    sys_text = first["content"] + ("\n\n" + tools_prompt(specs, compact) if specs else "")
    out = [{"role": "system", "content": sys_text}]
    last_tool_user = False
    for m in messages[1:]:
        role = m["role"]
        content = m.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content)
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except ValueError:
                    args = {}
                content += "\n<tool_call>\n" + json.dumps({"name": tc["function"]["name"], "arguments": args}) + "\n</tool_call>"
            out.append({"role": "assistant", "content": content.strip() or "(no text)"})
            last_tool_user = False
        elif role == "tool":
            block = "<tool_response>\n" + content + "\n</tool_response>"
            if last_tool_user:
                out[-1]["content"] += "\n" + block
            else:
                out.append({"role": "user", "content": block})
                last_tool_user = True
        else:
            out.append({"role": "user", "content": content})
            last_tool_user = False
    return out


# ---------------------------------------------------------------- parsing tool-call blocks
def _loads_lenient(text: str):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    for cand in (t, re.sub(r",\s*([}\]])", r"\1", t)):
        try:
            return json.loads(cand)
        except ValueError:
            pass
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        for cand in (m.group(0), re.sub(r",\s*([}\]])", r"\1", m.group(0))):
            try:
                return json.loads(cand)
            except ValueError:
                pass
    return None


def _coerce(v: str):
    v = v.strip("\n")
    try:
        return json.loads(v)
    except ValueError:
        return v


def parse_tool_block(block: str) -> list:
    """Return [{"name": str, "args": dict}] for one <tool_call> block (may hold several calls)."""
    t = block.strip()
    calls = []
    if "<function=" in t:
        for m in re.finditer(r"<function=([\w.\-]+)>(.*?)(?:</function>|$)", t, re.S):
            args = {pm.group(1): _coerce(pm.group(2)) for pm in re.finditer(r"<parameter=([\w.\-]+)>\s*(.*?)\s*</parameter>", m.group(2), re.S)}
            calls.append({"name": m.group(1), "args": args})
        return calls
    obj = _loads_lenient(t)
    for o in (obj if isinstance(obj, list) else [obj]):
        if not isinstance(o, dict):
            continue
        name = o.get("name") or o.get("function") or (o.get("function", {}) if isinstance(o.get("function"), dict) else None)
        if isinstance(name, dict):
            o, name = name, name.get("name")
        args = o.get("arguments", o.get("parameters", o.get("args", {})))
        if isinstance(args, str):
            args = _loads_lenient(args) or {}
        if name and isinstance(args, dict):
            calls.append({"name": str(name), "args": args})
    return calls


def parse_bare_calls(text: str, known: set) -> list:
    """Fallback: the whole reply is a JSON call object (some models skip the tags)."""
    t = text.strip()
    if not t or (t[0] not in "{[`"):
        return []
    return [c for c in parse_tool_block(t) if c["name"] in known]


class StreamSplitter:
    """Splits a token stream into visible text, <think> reasoning and <tool_call> blocks."""

    def __init__(self, on_text=None, on_reason=None):
        self.on_text, self.on_reason = on_text, on_reason
        self.buf, self.mode = "", "text"
        self.text, self.reasoning, self.tool_blocks = [], [], []

    def _emit(self, s, reasoning=False):
        if not s:
            return
        (self.reasoning if reasoning else self.text).append(s)
        cb = self.on_reason if reasoning else self.on_text
        if cb:
            cb(s)

    @staticmethod
    def _partial_suffix(buf: str, tags) -> int:
        best = 0
        for tag in tags:
            for k in range(1, len(tag)):
                if buf.endswith(tag[:k]):
                    best = max(best, k)
        return best

    def feed(self, s: str):
        self.buf += s
        while True:
            if self.mode == "text":
                hits = [(self.buf.find(t), t) for t in OPEN_TAGS if t in self.buf]
                if hits:
                    idx, tag = min(hits)
                    self._emit(self.buf[:idx])
                    self.buf, self.mode = self.buf[idx + len(tag):], OPEN_TAGS[tag]
                    continue
                keep = self._partial_suffix(self.buf, OPEN_TAGS)
                self._emit(self.buf[:len(self.buf) - keep])
                self.buf = self.buf[len(self.buf) - keep:]
                return
            close = CLOSE_TAGS[self.mode]
            if close in self.buf:
                body, self.buf = self.buf.split(close, 1)
                if self.mode == "think":
                    self._emit(body, reasoning=True)
                else:
                    self.tool_blocks.append(body)
                self.mode = "text"
                continue
            if self.mode == "think":
                keep = self._partial_suffix(self.buf, [close])
                self._emit(self.buf[:len(self.buf) - keep], reasoning=True)
                self.buf = self.buf[len(self.buf) - keep:]
            return

    def finish(self):
        if self.mode == "text":
            self._emit(self.buf)
        elif self.mode == "think":
            self._emit(self.buf, reasoning=True)
        elif self.buf.strip():
            self.tool_blocks.append(self.buf)   # model stopped at </tool_call> (stop token) or was cut off
        self.buf, self.mode = "", "text"


class TextToolClient:
    """Wraps any chat client that has no native tool calling; adds tools via the prompt and parses the calls back."""

    def __init__(self, inner, compact: bool = True):
        self.inner, self.compact = inner, compact
        self.native = False

    on_retry = property(lambda self: getattr(self.inner, "on_retry", None),
                        lambda self, v: setattr(self.inner, "on_retry", v))
    base_url = property(lambda self: self.inner.base_url)
    api_key = property(lambda self: self.inner.api_key)

    def list_models(self):
        return self.inner.list_models()

    def chat_stream(self, model, messages, tools=None, max_tokens=8192, temperature=0.3, on_text=None, on_reasoning=None):
        if not tools:
            return self.inner.chat_stream(model, messages, None, max_tokens, temperature, on_text, on_reasoning)
        sp = StreamSplitter(on_text, on_reasoning)
        r = self.inner.chat_stream(model, convert_messages(messages, tools, self.compact), None, max_tokens, temperature,
                                   sp.feed, on_reasoning)
        if not sp.text and not sp.tool_blocks and not sp.buf and r.get("content"):   # inner client didn't stream via callbacks
            sp.feed(r["content"])
        sp.finish()
        calls = []
        for b in sp.tool_blocks:
            calls += parse_tool_block(b)
        text = "".join(sp.text)
        if not calls:
            calls = parse_bare_calls(text, {s["function"]["name"] for s in tools})
        return {
            "content": "" if (calls and not sp.tool_blocks) else text.strip(),
            "reasoning": "".join(sp.reasoning) or r.get("reasoning", ""),
            "tool_calls": [{"id": "call_" + uuid.uuid4().hex[:9], "name": c["name"], "args": json.dumps(c["args"])} for c in calls],
            "finish_reason": "tool_calls" if calls else r.get("finish_reason"),
            "usage": r.get("usage"),
        }


class AutoToolClient:
    """Native tool calling first; permanently falls back to the text protocol if the server rejects/ignores tools."""

    def __init__(self, native, report=None):
        self.native, self.text = native, TextToolClient(native)
        self.mode, self.report = "native", report or (lambda m: None)

    on_retry = property(lambda self: getattr(self.native, "on_retry", None),
                        lambda self, v: setattr(self.native, "on_retry", v))
    base_url = property(lambda self: self.native.base_url)
    api_key = property(lambda self: self.native.api_key)

    def list_models(self):
        return self.native.list_models()

    def chat_stream(self, model, messages, tools=None, *a, **kw):
        from .client import NIMError
        if not tools or self.mode == "text":
            return (self.text if tools else self.native).chat_stream(model, messages, tools, *a, **kw)
        try:
            r = self.native.chat_stream(model, messages, tools, *a, **kw)
        except NIMError as e:
            if e.status in (400, 422, 500, 501):
                self.mode = "text"
                self.report("Server rejected native tool calling (%s); using prompt-based tool calling." % e.message[:80])
                return self.text.chat_stream(model, messages, tools, *a, **kw)
            raise
        if not r["tool_calls"] and "<tool_call>" in (r["content"] or ""):
            self.mode = "text"
            self.report("Model emits <tool_call> text; switching to prompt-based tool calling.")
            calls = []
            for b in re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", r["content"], re.S):
                calls += parse_tool_block(b)
            r["tool_calls"] = [{"id": "call_" + uuid.uuid4().hex[:9], "name": c["name"], "args": json.dumps(c["args"])} for c in calls]
            r["content"] = re.sub(r"<tool_call>.*?(</tool_call>|$)", "", r["content"], flags=re.S).strip()
        return r
