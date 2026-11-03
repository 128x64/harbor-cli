"""Mock NVIDIA NIM (OpenAI-compatible SSE) server used by the tests. Run standalone: python mock_nim.py PORT"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

SEEN = {"auth": [], "bodies": []}


def sse(h, chunks):
    h.send_response(200)
    h.send_header("Content-Type", "text/event-stream")
    h.end_headers()
    for c in chunks:
        h.wfile.write(("data: " + json.dumps(c) + "\n\n").encode())
        h.wfile.flush()
    h.wfile.write(b"data: [DONE]\n\n")


def delta(d, finish=None):
    return {"choices": [{"index": 0, "delta": d, "finish_reason": finish}]}


def call(idx, cid, name, args):
    return delta({"tool_calls": [{"index": idx, "id": cid, "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(args)}}]})


def steps_since_user(msgs):
    n = 0
    for m in reversed(msgs):
        if m["role"] == "user":
            break
        if m["role"] == "tool":
            n += 1
    return n


def root_user(msgs):
    for m in msgs[1:]:
        if m["role"] == "user":
            c = m["content"]
            return c if isinstance(c, str) else json.dumps(c)
    return ""


def first_user(msgs):
    for m in reversed(msgs):
        if m["role"] == "user":
            c = m["content"]
            return c if isinstance(c, str) else json.dumps(c)
    return ""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        SEEN["auth"].append(self.headers.get("Authorization"))
        body = json.dumps({"data": [{"id": "b/model"}, {"id": "a/model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        SEEN["auth"].append(self.headers.get("Authorization"))
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN["bodies"].append(body)
        msgs = body["messages"]
        sysmsg = msgs[0]["content"]
        user = first_user(msgs)
        step = steps_since_user(msgs)
        names = [t["function"]["name"] for t in body.get("tools", [])]
        if "worker sub-agent" in sysmsg:
            return sse(self, [delta({"content": "WORKER REPORT: done"}), delta({}, "stop")])
        if "research sub-agent" in sysmsg:
            return sse(self, [delta({"content": "SUB REPORT: found 3 files"}), delta({}, "stop")])
        root = root_user(msgs)
        if "SCENARIO:auto" in root:
            nudges = sum(1 for m in msgs if m["role"] == "user" and str(m["content"]).startswith("[auto]"))
            if nudges == 0:
                return sse(self, [delta({"content": "I think that's about it."}), delta({}, "stop")])
            if step == 0:
                return sse(self, [call(0, "call_w1", "write_file", {"path": "auto.txt", "content": "autonomous\n"}), delta({}, "tool_calls")])
            return sse(self, [call(0, "call_f1", "finish", {"summary": "auto summary: wrote auto.txt", "success": True}), delta({}, "tool_calls")])
        if "SCENARIO:verify" in root:
            if user.startswith("Verification command"):
                if step == 0:
                    return sse(self, [call(0, "call_v2", "write_file", {"path": "marker.txt", "content": "ok"}), delta({}, "tool_calls")])
                return sse(self, [delta({"content": "fixed it"}), delta({}, "stop")])
            if step == 0:
                return sse(self, [call(0, "call_v1", "write_file", {"path": "wrong.txt", "content": "x"}), delta({}, "tool_calls")])
            return sse(self, [delta({"content": "all done"}), delta({}, "stop")])
        if "SCENARIO:parallel" in root:
            if step == 0:
                return sse(self, [call(i, "call_p%d" % i, "slow", {"tag": str(i)}) for i in range(3)] + [delta({}, "tool_calls")])
            return sse(self, [delta({"content": "parallel done"}), delta({}, "stop")])
        if "SCENARIO:delegate" in root:
            if step == 0:
                return sse(self, [call(0, "call_d1", "delegate", {"description": "do work", "prompt": "build it"}), delta({}, "tool_calls")])
            return sse(self, [delta({"content": "delegated ok"}), delta({}, "stop")])
        if "SCENARIO:plan" in user:
            return sse(self, [delta({"content": "TOOLS:" + ",".join(names)}), delta({}, "stop")])
        if "SCENARIO:sub" in user:
            if step == 0:
                return sse(self, [call(0, "call_t1", "task", {"description": "look around", "prompt": "find files"}), delta({}, "tool_calls")])
            return sse(self, [delta({"content": "main done"}), delta({}, "stop")])
        if "SCENARIO:mcp" in user:
            if step == 0:
                return sse(self, [call(0, "call_m1", "mcp__mock__echo", {"text": "hi"}), delta({}, "tool_calls")])
            return sse(self, [delta({"content": "mcp done"}), delta({}, "stop")])
        if "SCENARIO:shell" in user:
            if step == 0:
                return sse(self, [call(0, "call_s1", "shell", {"command": "echo ran-ok"}), delta({}, "tool_calls")])
            return sse(self, [delta({"content": "shell finished"}), delta({}, "stop")])
        if "SCENARIO:slow" in user:
            import time
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for i in range(120):
                    self.wfile.write(("data: " + json.dumps(delta({"content": "tick %d\n" % i})) + "\n\n").encode())
                    self.wfile.flush()
                    time.sleep(0.1)
                self.wfile.write(b"data: [DONE]\n\n")
            except Exception:
                pass
            return
        # default "files" scenario
        if step == 0:
            return sse(self, [
                delta({"reasoning_content": "let me think"}),
                delta({"content": "I'll create a file.\n\n**Plan**: write `hello.py`.\n```python\nprint('x')\n```\n"}),
                delta({"tool_calls": [{"index": 0, "id": "call_a1", "type": "function", "function": {"name": "write_file", "arguments": ""}}]}),
                delta({"tool_calls": [{"index": 0, "function": {"arguments": "{\"path\": \"hello.py\", "}}]}),
                delta({"tool_calls": [{"index": 0, "function": {"arguments": "\"content\": \"print('hi from nim')\\n\"}"}}]}),
                call(1, "call_b2", "shell", {"command": "python3 hello.py"}),
                delta({}, "tool_calls")])
        if step == 2:
            return sse(self, [call(0, "call_c3", "edit_file", {"path": "hello.py", "old_string": "hi from nim", "new_string": "edited"}),
                              delta({}, "tool_calls")])
        return sse(self, [delta({"content": "All "}), delta({"content": "done."}), delta({}, "stop")])


def serve(port=0):
    srv = HTTPServer(("127.0.0.1", port), Handler)
    return srv


if __name__ == "__main__":
    s = serve(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    print(s.server_address[1], flush=True)
    s.serve_forever()
