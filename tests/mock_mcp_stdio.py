"""Tiny MCP server over stdio (newline-delimited JSON-RPC) for tests."""
import json
import sys

print("this line is noise, not JSON", flush=True)
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    if "id" not in msg:
        continue
    m, rid = msg["method"], msg["id"]
    if m == "initialize":
        res = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}, "resources": {}}, "serverInfo": {"name": "mock", "version": "1"}}
    elif m == "tools/list":
        res = {"tools": [{"name": "echo", "description": "Echo text back",
                          "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
                         {"name": "peek", "description": "read only", "annotations": {"readOnlyHint": True}, "inputSchema": {"type": "object"}}]}
    elif m == "tools/call":
        p = msg["params"]
        if p["name"] == "echo":
            res = {"content": [{"type": "text", "text": "echo: " + p["arguments"].get("text", "")}]}
        else:
            res = {"content": [{"type": "text", "text": "boom"}], "isError": True}
    elif m == "resources/list":
        res = {"resources": [{"uri": "mock://doc/1", "name": "Doc One"}]}
    elif m == "resources/read":
        res = {"contents": [{"uri": msg["params"]["uri"], "text": "resource body"}]}
    elif m == "ping":
        res = {}
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "nope"}}), flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": rid, "result": res}), flush=True)
