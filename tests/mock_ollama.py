"""Mock Ollama server (native /api/chat, NDJSON streaming) for the tests."""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SEEN = {"bodies": [], "auth": [], "generate": []}


def nd(h, objs):
    h.send_response(200)
    h.send_header("Content-Type", "application/x-ndjson")
    h.end_headers()
    for o in objs:
        h.wfile.write((json.dumps(o) + "\n").encode())
        h.wfile.flush()


def msg(content="", **kw):
    return {"model": "m", "created_at": "t", "message": dict({"role": "assistant", "content": content}, **kw), "done": False}


DONE = {"model": "m", "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop", "prompt_eval_count": 11, "eval_count": 7}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        SEEN["auth"].append(self.headers.get("Authorization"))
        if self.path == "/api/version":
            return self._json(200, {"version": "0.5.0"})
        if self.path == "/api/tags":
            return self._json(200, {"models": [{"name": "qwen2.5-coder:7b"}, {"name": "notools:1b"}]})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        SEEN["auth"].append(self.headers.get("Authorization"))
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/generate":                    # model preload: empty prompt
            SEEN["generate"].append(body)
            return self._json(200, {"model": body.get("model"), "done": True, "done_reason": "load", "response": ""})
        SEEN["bodies"].append(body)
        if self.path != "/api/chat":
            return self._json(404, {"error": "not found"})
        model, msgs, tools = body["model"], body["messages"], body.get("tools")
        if model == "missing:1b":
            return self._json(404, {"error": "model 'missing:1b' not found"})
        if tools and model == "notools:1b":
            return self._json(400, {"error": "registry.ollama.ai/library/notools:1b does not support tools"})
        if "<tools>" in msgs[0]["content"]:                       # prompt-based protocol
            if "<tool_response>" in msgs[-1]["content"]:
                return nd(self, [msg("text mode done"), DONE])
            return nd(self, [msg('ok <tool_call>{"name":"write_file",'), msg('"arguments":{"path":"oll.txt","content":"text"}}</tool_call>'), DONE])
        n_tool = 0
        for m in reversed(msgs):
            if m["role"] == "user":
                break
            if m["role"] == "tool":
                n_tool += 1
        if n_tool == 0:
            return nd(self, [msg("", thinking="considering"), msg("Writing it. "),
                             msg("", tool_calls=[{"function": {"name": "write_file", "arguments": {"path": "oll.txt", "content": "hello"}}}]), DONE])
        return nd(self, [msg("ollama "), msg("done"), DONE])


def serve(port=0):
    s = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    s.daemon_threads = True
    return s


if __name__ == "__main__":
    s = serve(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    print(s.server_address[1], flush=True)
    s.serve_forever()
