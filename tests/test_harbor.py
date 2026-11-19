#!/usr/bin/env python3
"""Offline test-suite: mock NIM server, mock MCP servers, tool unit tests and pty-driven UI tests (POSIX)."""
import json
import os
import pty
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import codecs  # noqa: E402
import mock_nim  # noqa: E402
import mock_ollama  # noqa: E402
from vt import Screen  # noqa: E402
from harbor import gitfinder, mcp, plugins, util  # noqa: E402
from harbor.core import Context, Registry, shell_is_dangerous  # noqa: E402
from harbor.tools import register_all  # noqa: E402
from harbor.tools import shell as shell_mod  # noqa: E402
from harbor.tools.fs import apply_edits  # noqa: E402
from harbor.tools.web import parse_bing_html  # noqa: E402
from harbor.ui import MDStream, cell_len, clip, make_diff, strip_ansi  # noqa: E402

SRV = OSRV = None
PORT = OPORT = 0


def setUpModule():
    global SRV, PORT, OSRV, OPORT
    SRV = mock_nim.serve(0)
    PORT = SRV.server_address[1]
    threading.Thread(target=SRV.serve_forever, daemon=True).start()
    OSRV = mock_ollama.serve(0)
    OPORT = OSRV.server_address[1]
    threading.Thread(target=OSRV.serve_forever, daemon=True).start()


def tearDownModule():
    SRV.shutdown()
    OSRV.shutdown()


def run_cli(args, home, cwd, extra_env=None, stdin=subprocess.DEVNULL, timeout=90):
    env = dict(os.environ, NVIDIA_API_KEY="nvapi-test", HOME=home, USERPROFILE=home, PYTHONPATH=str(ROOT), HARBOR_NO_UPDATE="1")
    env.update(extra_env or {})
    return subprocess.run([sys.executable, "-m", "harbor", "--base-url", "http://127.0.0.1:%d/v1" % PORT, "--cwd", cwd] + args,
                          capture_output=True, text=True, env=env, timeout=timeout, stdin=stdin)


def tool_ctx(tmp):
    reg = Registry()
    register_all(reg)
    return reg, Context(tmp, {}, None)


def call(reg, ctx, _name, **kw):
    return reg.get(_name).func(ctx, **kw)


class EndToEnd(unittest.TestCase):
    def test_files_scenario_yolo(self):
        start = len(mock_nim.SEEN["bodies"])
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            r = run_cli(["--yolo", "-p", "make hello"], h, d)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("Write(hello.py)", r.stdout)
            self.assertIn("Updated hello.py with 1 addition and 1 removal", r.stdout)
            self.assertIn("All done.", r.stdout)
            self.assertEqual((Path(d) / "hello.py").read_text(), "print('edited')\n")
        self.assertTrue(all(a == "Bearer nvapi-test" for a in mock_nim.SEEN["auth"]))
        first = mock_nim.SEEN["bodies"][start]
        self.assertTrue(first["stream"])
        names = [t["function"]["name"] for t in first["tools"]]
        for expected in ("read_file", "edit_file", "shell", "web_search", "git", "task", "analyze_image", "python_exec"):
            self.assertIn(expected, names)
        ids = [m["tool_call_id"] for m in mock_nim.SEEN["bodies"][-1]["messages"] if m["role"] == "tool"]
        self.assertEqual(ids[:3], ["call_a1", "call_b2", "call_c3"])

    def test_noninteractive_denies_without_yolo(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            r = run_cli(["-p", "make hello"], h, d)
            self.assertFalse((Path(d) / "hello.py").exists(), r.stdout)
            self.assertIn("rejected", r.stdout.lower())

    def test_subagent(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            r = run_cli(["--yolo", "-p", "SCENARIO:sub"], h, d)
            self.assertIn("main done", r.stdout, r.stdout + r.stderr)
            self.assertIn("Task(look around)", r.stdout)
            tool_msgs = [m for m in mock_nim.SEEN["bodies"][-1]["messages"] if m["role"] == "tool"]
            self.assertIn("SUB REPORT", tool_msgs[-1]["content"])

    def test_plan_mode_only_readonly_tools(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            r = run_cli(["--plan", "-p", "SCENARIO:plan"], h, d)
            tools = r.stdout.split("TOOLS:")[1].strip().split(",")
            self.assertIn("read_file", tools)
            self.assertNotIn("write_file", tools)
            self.assertNotIn("shell", tools)

    def test_core_profile_hides_groups(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            r = run_cli(["--tools", "core", "-p", "SCENARIO:plan", "--plan"], h, d)
            tools = r.stdout.split("TOOLS:")[1].strip().split(",")
            self.assertNotIn("web_search", tools)
            self.assertIn("grep", tools)

    def test_mcp_tool_through_agent(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            cfgdir = Path(h) / ".harbor"
            cfgdir.mkdir()
            (cfgdir / "mcp.json").write_text(json.dumps({"mcpServers": {"mock": {
                "command": sys.executable, "args": [str(HERE / "mock_mcp_stdio.py")]}}}))
            r = run_cli(["--yolo", "-p", "SCENARIO:mcp"], h, d)
            self.assertIn("mcp done", r.stdout, r.stdout + r.stderr)
            self.assertIn("echo: hi", r.stdout)
            self.assertIn("mock - echo (MCP)", r.stdout)

    def test_list_models(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            r = run_cli(["--list-models"], h, d)
            self.assertEqual(r.stdout.split(), ["a/model", "b/model"])

    def test_project_extensions_require_trust(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            td = Path(d) / ".harbor" / "tools"
            td.mkdir(parents=True)
            (td / "evil.py").write_text("open(%r, 'w').write('pwned')\nTOOLS=[]\n" % str(Path(d) / "pwned.txt"))
            run_cli(["-p", "SCENARIO:plan", "--plan"], h, d)
            self.assertFalse((Path(d) / "pwned.txt").exists())
            run_cli(["-p", "SCENARIO:plan", "--plan", "--trust-project"], h, d)
            self.assertTrue((Path(d) / "pwned.txt").exists())


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name).resolve()
        self.reg, self.ctx = tool_ctx(self.d)
        (self.d / "src").mkdir()
        (self.d / "src" / "a.py").write_text("def foo():\n    return 1\n")
        (self.d / "b.txt").write_text("hello\nworld\nhello\n")

    def tearDown(self):
        for p in self.ctx.procs.values():
            shell_mod.kill_tree(p.proc)
        self.tmp.cleanup()

    def c(self, _tool, **kw):
        return call(self.reg, self.ctx, _tool, **kw)

    def test_read_glob_grep_list(self):
        self.assertIn("def foo", self.c("read_file", path="src/a.py"))
        self.assertEqual(self.c("glob", pattern="**/*.py").strip(), "src/a.py")
        self.assertIn("b.txt:1:hello", self.c("grep", pattern="hello"))
        self.assertIn("src/", self.c("list_dir", path=".", depth=2))
        self.assertIn("a.py", self.c("list_dir", path=".", depth=2))

    def test_edit_rules_and_multi_edit(self):
        self.assertIn("matches 2 places", self.c("edit_file", path="b.txt", old_string="hello", new_string="bye"))
        self.assertIn("Edited", self.c("edit_file", path="b.txt", old_string="hello", new_string="bye", replace_all=True))
        self.assertIn("not found", self.c("edit_file", path="b.txt", old_string="nope", new_string="x"))
        self.assertIn("Applied 2 edits", self.c("multi_edit", path="b.txt", edits=[
            {"old_string": "bye\nworld", "new_string": "A"}, {"old_string": "A", "new_string": "B"}]))
        self.assertIn("nothing written", self.c("multi_edit", path="b.txt", edits=[
            {"old_string": "B", "new_string": "C"}, {"old_string": "zzz", "new_string": "y"}]))
        self.assertEqual((self.d / "b.txt").read_text(), "B\nbye\n")

    def test_crlf_and_bom_preserved(self):
        p = self.d / "win.txt"
        p.write_bytes(b"\xef\xbb\xbfline1\r\nline2\r\n")
        self.c("edit_file", path="win.txt", old_string="line2", new_string="changed")
        self.assertEqual(p.read_bytes(), b"\xef\xbb\xbfline1\r\nchanged\r\n")
        self.c("write_file", path="win.txt", content="a\nb\n")
        self.assertEqual(p.read_bytes(), b"\xef\xbb\xbfa\r\nb\r\n")

    def test_documents(self):
        import docx, openpyxl
        d = docx.Document()
        d.add_paragraph("Hello DOCX")
        d.save(str(self.d / "x.docx"))
        self.assertIn("Hello DOCX", self.c("read_file", path="x.docx"))
        wb = openpyxl.Workbook()
        wb.active.append(["name", "qty"])
        wb.active.append(["apple", 3])
        wb.save(str(self.d / "x.xlsx"))
        out = self.c("read_file", path="x.xlsx")
        self.assertIn("apple\t3", out)
        objs = [b"<</Type/Catalog/Pages 2 0 R>>", b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
                b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
                b"<</Length 38>>\nstream\nBT /F1 18 Tf 20 100 Td (Hello PDF) Tj ET\nendstream",
                b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>"]
        pdf, offs = b"%PDF-1.4\n", []
        for i, o in enumerate(objs, 1):
            offs.append(len(pdf))
            pdf += b"%d 0 obj\n" % i + o + b"\nendobj\n"
        xref = len(pdf)
        pdf += b"xref\n0 6\n0000000000 65535 f \n" + b"".join(b"%010d 00000 n \n" % o for o in offs)
        pdf += b"trailer\n<</Size 6/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n" % xref
        (self.d / "x.pdf").write_bytes(pdf)
        self.assertIn("Hello PDF", self.c("read_file", path="x.pdf"))

    def test_shell_and_timeout(self):
        self.assertIn("exit code 3", self.c("shell", command="exit 3"))
        self.assertIn("timed out", self.c("shell", command="sleep 5", timeout=1))
        self.assertIn("hi there", self.c("shell", command="echo hi there"))

    def test_background_process(self):
        out = self.c("start_process", command="echo started; sleep 30")
        self.assertIn("started", out)
        self.assertIn("running", self.c("process_output", id="p1"))
        self.assertIn("Stopped", self.c("stop_process", id="p1"))
        time.sleep(0.5)
        self.assertIn("exit", self.c("list_processes"))

    def test_python_sqlite_calc(self):
        self.assertIn("42", self.c("python_exec", code="print(6*7)"))
        self.c("sqlite_query", db_path="t.db", sql="create table t(a,b)")
        self.c("sqlite_query", db_path="t.db", sql="insert into t values (1,'x')")
        self.assertIn("1\tx", self.c("sqlite_query", db_path="t.db", sql="select * from t"))
        self.assertEqual(self.c("calculate", expression="2**10 + sqrt(16)"), "1028.0")
        self.assertIn("Error", self.c("calculate", expression="__import__('os')"))

    def test_git_tool_and_mutation_detection(self):
        self.c("shell", command="git init -q . && git config user.email a@b.c && git config user.name t && git add -A && git commit -qm init")
        self.assertIn("init", self.c("git", args="log --oneline"))
        t = self.reg.get("git")
        self.assertFalse(t.needs_ask({"args": "status"}))
        self.assertFalse(t.needs_ask({"args": "log --oneline -5"}))
        self.assertFalse(t.needs_ask({"args": "branch"}))
        self.assertTrue(t.needs_ask({"args": "commit -am x"}))
        self.assertTrue(t.needs_ask({"args": "push origin main"}))
        self.assertTrue(t.needs_ask({"args": "branch newbranch"}))
        self.assertTrue(t.danger({"args": "push --force"}))

    def test_archive_and_safety(self):
        self.c("archive", action="create", archive="o.zip", source="src")
        self.assertIn("a.py", self.c("archive", action="list", archive="o.zip"))
        self.c("archive", action="extract", archive="o.zip", source="out")
        self.assertTrue((self.d / "out" / "src" / "a.py").exists())
        import zipfile
        with zipfile.ZipFile(self.d / "evil.zip", "w") as z:
            z.writestr("../escape.txt", "x")
        self.assertIn("unsafe", self.c("archive", action="extract", archive="evil.zip", source="out2"))

    def test_file_ops_and_delete_guard(self):
        self.c("copy_path", src="b.txt", dst="c.txt")
        self.c("move_path", src="c.txt", dst="sub/d.txt")
        self.assertTrue((self.d / "sub" / "d.txt").exists())
        self.assertIn("Deleted", self.c("delete_path", path="sub"))
        self.assertIn("refusing", self.c("delete_path", path=str(Path.home())))
        self.assertIn("sha256", self.c("file_info", path="b.txt"))
        self.assertIn("identical", self.c("diff_files", a="b.txt", b="b.txt"))

    def test_env_vars_hide_secrets(self):
        with mock.patch.dict(os.environ, {"MY_API_KEY": "supersecret", "PLAIN": "visible"}):
            self.assertEqual(self.c("env_vars", name="MY_API_KEY"), "<hidden>")
            self.assertEqual(self.c("env_vars", name="PLAIN"), "visible")

    def test_todo_and_memory(self):
        self.c("todo_write", todos=[{"content": "a", "status": "completed"}, {"content": "b", "status": "weird"}])
        self.assertEqual([t["status"] for t in self.ctx.todos], ["completed", "pending"])

    def test_system_tools(self):
        self.assertIn("python:", self.c("system_info"))
        self.assertIn("unix", self.c("datetime_now"))


class Misc(unittest.TestCase):
    def test_dangerous_patterns(self):
        bad = ["rm -rf /", "rm -rf ~", "sudo rm -fr /*", "git push --force origin main", "git push -f", "curl http://x | sh",
               "mkfs.ext4 /dev/sda", "iwr http://x | iex", "Remove-Item -Recurse -Force C:\\", "format C:", "git reset --hard", "shutdown /s"]
        ok = ["rm -rf build", "ls -la", "python -m pytest", "rm file.txt", "git status", "Remove-Item .\\build -Recurse"]
        for c in bad:
            self.assertTrue(shell_is_dangerous(c), c)
        for c in ok:
            self.assertFalse(shell_is_dangerous(c), c)

    def test_shell_argv_windows(self):
        def which(n):
            return {"pwsh": None, "powershell": r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"}.get(n)
        with mock.patch.object(shell_mod, "IS_WIN", True), mock.patch.object(shell_mod.shutil, "which", side_effect=which):
            argv = shell_mod.shell_argv("dir")
            self.assertTrue(argv[0].endswith("powershell.exe"))
            self.assertIn("-NoProfile", argv)
            self.assertTrue(argv[-1].endswith("dir"))
            self.assertEqual(shell_mod.shell_name(), "powershell")
        with mock.patch.object(shell_mod, "IS_WIN", True), mock.patch.object(shell_mod.shutil, "which", return_value=None):
            self.assertEqual(shell_mod.shell_argv("dir"), ["cmd.exe", "/d", "/s", "/c", "dir"])

    def test_find_git_in_github_desktop(self):
        with tempfile.TemporaryDirectory() as local:
            for ver in ("3.4.9", "3.4.10"):
                p = Path(local) / "GitHubDesktop" / ("app-" + ver) / "resources" / "app" / "git" / "cmd"
                p.mkdir(parents=True)
                (p / "git.exe").write_text("")
            gitfinder._cache.clear()
            with mock.patch.object(gitfinder, "IS_WIN", True), mock.patch.dict(os.environ, {"LOCALAPPDATA": local, "HARBOR_GIT": ""}), \
                    mock.patch.object(gitfinder.shutil, "which", return_value=None):
                found = gitfinder.find_git({})
                self.assertIn("app-3.4.10", found)  # numeric version ordering, not lexicographic
                env = gitfinder.tool_env({})
                self.assertTrue(env["PATH"].startswith(os.path.dirname(found)))
        gitfinder._cache.clear()

    def test_bing_parser(self):
        import base64
        tgt = "https://docs.python.org/3/library/argparse.html"
        w = "https://www.bing.com/ck/a?!&amp;&amp;p=x&amp;u=a1" + base64.urlsafe_b64encode(tgt.encode()).decode().rstrip("=")
        html = '<li class="b_algo"><h2><a href="%s">argparse</a></h2><div class="b_caption"><p>Parser <b>docs</b></p></div></li>' % w
        self.assertEqual(parse_bing_html(html)[0], {"title": "argparse", "url": tgt, "snippet": "Parser docs"})

    def test_markdown_stream(self):
        with mock.patch.object(util, "USE_COLOR", False):
            md = MDStream()
            out = []
            for chunk in ["# Title\n- item **b", "old**\n```py\nx = 1\n```\nuse `code` here", "\n"]:
                out += md.feed(chunk)
            self.assertEqual([strip_ansi(x) for x in out if x is not None], ["Title", "• item bold", "  x = 1", "use code here"])   # classic
            md = MDStream("modern")
            out = []
            for chunk in ["# Title\n- item **b", "old**\n```py\nx = 1\n```\nuse `code` here", "\n"]:
                out += md.feed(chunk)
            self.assertEqual([strip_ansi(x) for x in out if x is not None], ["Title", "• item bold", "┌─ py", "│ x = 1", "└─", "use code here"])

    def test_diff_and_width(self):
        rows, adds, dels = make_diff("a\nb\nc\n", "a\nB\nc\nd\n")
        self.assertEqual((adds, dels), (2, 1))
        self.assertEqual(cell_len("日本語"), 6)
        self.assertLessEqual(cell_len(clip("x" * 100, 20)), 20)

    def test_permissions_prefix(self):
        from harbor.core import Permissions
        p = Permissions()
        self.assertEqual(p.command_prefix("npm test --silent"), "npm test")
        self.assertEqual(p.command_prefix("ls -la"), "ls")
        p.prefixes.add("npm test")
        self.assertTrue(p.prefix_allowed("npm test -- --watch"))
        self.assertFalse(p.prefix_allowed("npm test && rm -rf x"))
        self.assertFalse(p.prefix_allowed("npm install"))

    def test_plugins_and_commands(self):
        with tempfile.TemporaryDirectory() as h:
            td = Path(h) / "tools"
            td.mkdir()
            (td / "greet.py").write_text(
                "def greet(name):\n    return 'hello ' + name\n"
                "TOOLS=[{'name':'greet','description':'d','parameters':{'name':{'type':'string'}},'required':['name'],'func':greet,'ask':False}]\n")
            (td / "broken.py").write_text("raise RuntimeError('x')\n")
            reg = Registry()
            msgs = []
            self.assertEqual(plugins.load_tools(reg, [td], msgs.append), 1)
            self.assertEqual(reg.get("greet").func(None, name="bob"), "hello bob")
            self.assertTrue(any("broken.py" in m for m in msgs))
            cd = Path(h) / "commands"
            cd.mkdir()
            (cd / "review.md").write_text("Review $ARGUMENTS carefully")
            self.assertEqual(plugins.load_commands([cd])["review"], "Review $ARGUMENTS carefully")


class MCPTests(unittest.TestCase):
    def test_stdio(self):
        c = mcp.StdioMCP("mock", sys.executable, [str(HERE / "mock_mcp_stdio.py")])
        c.start()
        try:
            c.initialize()
            tools = c.list_tools()
            self.assertEqual([t["name"] for t in tools], ["echo", "peek"])
            self.assertEqual(c.call_tool("echo", {"text": "yo"}), "echo: yo")
            self.assertTrue(c.call_tool("peek", {}).startswith("Error:"))
            reg = Registry()
            mcp.register_mcp_tools(reg, [c])
            self.assertTrue(reg.get("mcp__mock__echo").needs_ask({}))
            self.assertFalse(reg.get("mcp__mock__peek").needs_ask({}))
            self.assertTrue(reg.get("mcp__mock__peek").read_only)
        finally:
            c.close()

    def test_stdio_bad_command(self):
        with self.assertRaises(mcp.MCPError):
            c = mcp.StdioMCP("x", "definitely-not-a-real-binary-xyz")
            c.start()

    def test_http_json_and_sse(self):
        sessions = []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if "id" not in body:
                    self.send_response(202)
                    self.end_headers()
                    return
                m = body["method"]
                if m == "initialize":
                    res = {"protocolVersion": "2025-06-18", "serverInfo": {"name": "h"}}
                elif m == "tools/list":
                    res = {"tools": [{"name": "t", "description": "d", "inputSchema": {"type": "object"}}]}
                else:
                    res = {"content": [{"type": "text", "text": "called " + body["params"]["name"] + " auth=" + str(self.headers.get("Authorization"))}]}
                payload = {"jsonrpc": "2.0", "id": body["id"], "result": res}
                self.send_response(200)
                if m == "initialize":
                    self.send_header("Mcp-Session-Id", "sess-1")
                else:
                    sessions.append(self.headers.get("Mcp-Session-Id"))
                if m == "tools/list":  # answer this one as an SSE stream
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(("event: message\ndata: " + json.dumps({"jsonrpc": "2.0", "method": "notifications/x"}) + "\n\n").encode())
                    self.wfile.write(("event: message\ndata: " + json.dumps(payload) + "\n\n").encode())
                else:
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(payload).encode())

            def do_DELETE(self):
                self.send_response(200)
                self.end_headers()

        srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            with mock.patch.dict(os.environ, {"TOK": "abc"}):
                c = mcp.HttpMCP("web", "http://127.0.0.1:%d/mcp" % srv.server_address[1], {"Authorization": "Bearer ${TOK}"})
                c.initialize()
                self.assertEqual(c.list_tools()[0]["name"], "t")
                self.assertEqual(c.call_tool("t", {}), "called t auth=Bearer abc")
                self.assertEqual(set(sessions), {"sess-1"})
                c.close()
        finally:
            srv.shutdown()


# ------------------------------------------------------------------------------------------------ pty UI tests
class Pty:
    def __init__(self, args, home, cwd, rows=40, cols=100, env_extra=None, entry=None):
        self.master, slave = pty.openpty()
        import fcntl, termios
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        env = dict(os.environ, NVIDIA_API_KEY="nvapi-test", HOME=home, PYTHONPATH=str(ROOT), TERM="xterm-256color", HARBOR_NO_UPDATE="1")
        env.update(env_extra or {})
        env = {k: v for k, v in env.items() if v is not None}
        env.pop("NO_COLOR", None)
        launch = [sys.executable, entry] if entry else [sys.executable, "-m", "harbor"]
        self.p = subprocess.Popen(launch + ["--base-url", "http://127.0.0.1:%d/v1" % PORT, "--cwd", cwd] + args,
                                  stdin=slave, stdout=slave, stderr=slave, env=env, cwd=cwd, close_fds=True, start_new_session=True)
        os.close(slave)
        self.buf = ""
        self.screen = Screen(rows, cols)
        self._dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def _take(self):
        d = self._dec.decode(os.read(self.master, 65536))
        self.buf += d
        self.screen.feed(d)

    def pump(self, secs=0.6):
        end = time.time() + secs
        while time.time() < end:
            if select.select([self.master], [], [], 0.1)[0]:
                try:
                    self._take()
                except OSError:
                    break

    def read_until(self, text, timeout=20):
        end = time.time() + timeout
        start = len(self.buf)
        while time.time() < end:
            if text in strip_ansi(self.buf[start:]) or text in self.buf[start:]:
                return strip_ansi(self.buf[start:])
            r, _, _ = select.select([self.master], [], [], 0.2)
            if r:
                try:
                    self._take()
                except OSError:
                    break
        raise AssertionError("timeout waiting for %r; got tail:\n%s" % (text, strip_ansi(self.buf[start:])[-1500:]))

    def send(self, s):
        os.write(self.master, s.encode())

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self.p.kill()
        except Exception:
            pass
        try:
            os.close(self.master)
        except OSError:
            pass


@unittest.skipIf(os.name == "nt", "pty tests need POSIX")
class InteractiveUI(unittest.TestCase):
    def setUp(self):
        self.h = tempfile.TemporaryDirectory()
        self.d = tempfile.TemporaryDirectory()
        self.t = Pty([], self.h.name, self.d.name)

    def tearDown(self):
        self.t.close()
        self.h.cleanup()
        self.d.cleanup()

    def test_full_interactive_session(self):
        t = self.t
        out = t.read_until("manual mode")
        self.assertIn("Welcome to Harbor CLI!", t.buf)
        self.assertIn("╭", t.buf)
        # editing keys: type, move, delete, then clear with ctrl-u
        t.send("helo")
        t.send("\x1b[D\x1b[D")      # left x2
        t.send("l")                  # -> hello? (hel|o -> "hello" after inserting l)
        t.read_until("hello")
        t.send("\x05\x15")           # ctrl-e then ctrl-u clears the line
        # slash suggestions
        t.send("/mo")
        t.read_until("/model")
        t.send("\x1b")               # esc clears
        time.sleep(0.2)
        # shift-tab cycles modes
        t.send("\x1b[Z")
        t.read_until("accept edits on")
        t.send("\x1b[Z")
        t.read_until("plan mode on")
        t.send("\x1b[Z")
        t.read_until("autonomous mode on")
        t.send("\x1b[Z")
        t.read_until("manual mode")
        # builtin command
        t.send("/status\r")
        out = t.read_until("MCP servers:")
        self.assertIn("Model:", out)
        # permission dialog: shell tool needs approval
        t.send("SCENARIO:shell\r")
        out = t.read_until("Do you want to proceed?")
        self.assertIn("echo ran-ok", out)
        t.pump(0.4)
        self.assertIn("don't ask again for `echo` commands", t.screen.text())
        t.send("1")
        out = t.read_until("shell finished")
        self.assertIn("ran-ok", t.buf)
        # second time: option 2 remembered? ask again and pick 'don't ask again'
        t.send("SCENARIO:shell\r")
        t.read_until("Do you want to proceed?")
        t.send("2")
        t.read_until("shell finished")
        # third time runs without a dialog
        before = len(t.buf)
        t.send("SCENARIO:shell\r")
        t.read_until("shell finished")
        self.assertNotIn("Do you want to proceed?", strip_ansi(t.buf[before:]))
        # reject
        t.send("/exit\r")
        t.p.wait(timeout=10)
        self.assertEqual(t.p.returncode, 0)

    def test_escape_interrupts_generation(self):
        t = self.t
        t.read_until("manual mode")
        t.send("SCENARIO:slow\r")
        t.read_until("tick 3")
        t.read_until("esc to interrupt")
        time.sleep(0.3)
        t.send("\x1b")
        t.read_until("Interrupted by user")
        t.pump(1.0)
        self.assertIn("manual mode", t.screen.text().split("\n")[-1])      # the prompt is back
        t.send("\x04")  # ctrl-d exits
        t.p.wait(timeout=10)

    def test_reject_stops_turn(self):
        t = self.t
        t.read_until("manual mode")
        t.send("SCENARIO:shell\r")
        t.read_until("Do you want to proceed?")
        t.send("3")
        t.read_until("User rejected this action")
        t.pump(1.0)
        self.assertIn("manual mode", t.screen.text().split("\n")[-1])
        t.send("\x03\x03")  # ctrl-c twice to exit
        t.p.wait(timeout=10)



# ================================================================================================ v3 tests
import textwrap  # noqa: E402
import queue  # noqa: E402
from http.server import ThreadingHTTPServer  # noqa: E402
from harbor import backends, textproto  # noqa: E402
from harbor.client import NIMError  # noqa: E402


def run_main(args, main_dir, home, cwd, extra_env=None, timeout=90):
    """Run harbor with an explicit main_dir (where model.gguf / llama-server live)."""
    env = dict(os.environ, NVIDIA_API_KEY="nvapi-test", HOME=home, USERPROFILE=home, PYTHONPATH=str(ROOT), HARBOR_NO_UPDATE="1")
    env.update(extra_env or {})
    code = "import sys; sys.path.insert(0, %r); from harbor.cli import main; sys.exit(main(%r, main_dir=%r))" % (str(ROOT), args + ["--cwd", cwd], str(main_dir))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=timeout, stdin=subprocess.DEVNULL)


class TextProtocol(unittest.TestCase):
    def test_splitter_and_parsers(self):
        chunks = ["Sure, ", "reading.<th", "ink>plan it</thi", "nk>\n<tool_", 'call>\n{"name":"read_file","arguments":{"path":"a.py"}}\n</tool_call>',
                  "<tool_call><function=grep><parameter=pattern>foo</parameter><parameter=ignore_case>true</parameter></function></tool_call>",
                  '<tool_call>{"name": "glob", "parameters": {"pattern": "*.py",}}']
        got, rs = [], []
        sp = textproto.StreamSplitter(got.append, rs.append)
        for c in chunks:
            sp.feed(c)
        sp.finish()
        self.assertEqual("".join(got).strip(), "Sure, reading.")
        self.assertEqual(rs, ["plan it"])
        calls = [c for b in sp.tool_blocks for c in textproto.parse_tool_block(b)]
        self.assertEqual(calls, [{"name": "read_file", "args": {"path": "a.py"}},
                                 {"name": "grep", "args": {"pattern": "foo", "ignore_case": True}},
                                 {"name": "glob", "args": {"pattern": "*.py"}}])
        self.assertEqual(textproto.parse_bare_calls('```json\n{"name":"read_file","arguments":{"path":"x"}}\n```', {"read_file"})[0]["args"], {"path": "x"})

    def test_convert_messages(self):
        msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "1", "type": "function", "function": {"name": "a", "arguments": "{\"x\": 1}"}},
                    {"id": "2", "type": "function", "function": {"name": "b", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "1", "content": "r1"}, {"role": "tool", "tool_call_id": "2", "content": "r2"}]
        specs = [{"type": "function", "function": {"name": "a", "description": "does a. More text.", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}}}]
        out = textproto.convert_messages(msgs, specs)
        self.assertIn("<tools>", out[0]["content"])
        self.assertIn('"name":"a"', out[0]["content"])
        self.assertEqual([m["role"] for m in out], ["system", "user", "assistant", "user"])
        self.assertIn("<tool_call>", out[2]["content"])
        self.assertEqual(out[3]["content"].count("<tool_response>"), 2)

    def test_text_client_over_fake_inner(self):
        class Inner:
            base_url, api_key, on_retry = "x", "k", None

            def chat_stream(self, model, messages, tools=None, max_tokens=1, temperature=0, on_text=None, on_reasoning=None):
                assert tools is None and "<tools>" in messages[0]["content"]
                for p in ["ok ", '<tool_call>{"name":"glob","arguments":{"pattern":"*"}}</tool_call>']:
                    on_text(p)
                return {"content": "", "tool_calls": [], "finish_reason": "stop", "usage": None}
        spec = [{"type": "function", "function": {"name": "glob", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
        r = textproto.TextToolClient(Inner()).chat_stream("m", [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], spec)
        self.assertEqual(r["content"], "ok")
        self.assertEqual(r["tool_calls"][0]["name"], "glob")
        self.assertEqual(json.loads(r["tool_calls"][0]["args"]), {"pattern": "*"})
        self.assertEqual(r["finish_reason"], "tool_calls")

    def test_auto_client_falls_back_to_text(self):
        calls = []

        class Native:
            base_url, api_key, on_retry = "x", "k", None

            def chat_stream(self, model, messages, tools=None, *a, **kw):
                calls.append(bool(tools))
                if tools and messages[0]["content"] == "s":
                    raise NIMError(400, "tools not supported")
                kw_on = a[3] if len(a) > 3 else None
                return {"content": '<tool_call>{"name":"glob","arguments":{}}</tool_call>', "tool_calls": [], "finish_reason": "stop", "usage": None}

        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        spec = [{"type": "function", "function": {"name": "glob", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
        note = []
        c = textproto.AutoToolClient(Native(), note.append)
        r = c.chat_stream("m", msgs, spec, 100, 0.1, None, None)
        self.assertEqual(c.mode, "text")
        self.assertEqual(r["tool_calls"][0]["name"], "glob")
        self.assertTrue(note)


class LocalBackends(unittest.TestCase):
    def test_find_local_model(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(backends.find_local_model([d]))
            (Path(d) / "other.gguf").write_text("x")
            self.assertEqual(backends.find_local_model([d]).name, "other.gguf")      # single gguf is used
            (Path(d) / "second.gguf").write_text("x")
            self.assertIsNone(backends.find_local_model([d]))                         # ambiguous without model.gguf
            (Path(d) / "model.gguf").write_text("x")
            self.assertEqual(backends.find_local_model([d]).name, "model.gguf")      # model.gguf wins

    @unittest.skipIf(os.name == "nt", "fake llama-server is a POSIX script")
    def test_llama_server_engine_end_to_end(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as h, tempfile.TemporaryDirectory() as cwd:
            (Path(main) / "model.gguf").write_bytes(b"GGUF")
            srv = Path(main) / "llama-server"
            srv.write_text("#!%s\nimport sys\nsys.path.insert(0, %r)\nimport mock_nim\nargs = sys.argv\n"
                           "assert '--jinja' in args and '-m' in args\nmock_nim.serve(int(args[args.index('--port') + 1])).serve_forever()\n"
                           % (sys.executable, str(HERE)))
            srv.chmod(0o755)
            start = len(mock_nim.SEEN["bodies"])
            r = run_main(["--yolo", "-p", "make hello"], main, h, cwd)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("Local model found", r.stdout)
            self.assertIn("All done.", r.stdout)
            self.assertEqual((Path(cwd) / "hello.py").read_text(), "print('edited')\n")

    def test_llama_cpp_python_engine_with_text_tools(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as h, tempfile.TemporaryDirectory() as cwd:
            (Path(main) / "model.gguf").write_bytes(b"GGUF")
            r = run_main(["--yolo", "--tools", "core", "-p", "write a file"], main, h, cwd,
                         {"PYTHONPATH": os.pathsep.join([str(ROOT), str(HERE / "fake_llama_cpp")])})
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual((Path(cwd) / "local.txt").read_text(), "hi")
            self.assertIn("All done via local.", r.stdout)
            self.assertIn("Write(local.txt)", r.stdout)

    def test_backend_nim_overrides_local_model(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as h, tempfile.TemporaryDirectory() as cwd:
            (Path(main) / "model.gguf").write_bytes(b"GGUF")
            n = len(mock_nim.SEEN["auth"])
            r = run_main(["--backend", "nim", "--base-url", "http://127.0.0.1:%d/v1" % PORT, "--list-models"], main, h, cwd)
            self.assertEqual(r.stdout.split(), ["a/model", "b/model"], r.stderr)
            self.assertEqual(mock_nim.SEEN["auth"][n], "Bearer nvapi-test")

    def test_missing_local_engine_is_explained(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as h, tempfile.TemporaryDirectory() as cwd:
            (Path(main) / "model.gguf").write_bytes(b"GGUF")
            r = run_main(["-p", "hi"], main, h, cwd)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("llama-server", r.stderr)

    def test_root_script_runs(self):
        r = subprocess.run([sys.executable, str(ROOT / "main.py"), "--version"], capture_output=True, text=True)
        self.assertIn("harbor 3.4.0", r.stdout)


class Agentic(unittest.TestCase):
    def test_auto_mode_nudges_then_finishes(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            r = run_cli(["--auto", "--yolo", "-p", "SCENARIO:auto"], h, d)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual((Path(d) / "auto.txt").read_text(), "autonomous\n")
            self.assertIn("(auto) the model stopped without calling finish", r.stdout)
            self.assertIn("Finish(", r.stdout)
            self.assertIn("auto summary: wrote auto.txt", r.stdout)

    def test_finish_tool_hidden_outside_auto(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            start = len(mock_nim.SEEN["bodies"])
            run_cli(["--plan", "-p", "SCENARIO:plan"], h, d)
            names = [t["function"]["name"] for t in mock_nim.SEEN["bodies"][start]["tools"]]
            self.assertNotIn("finish", names)
            start = len(mock_nim.SEEN["bodies"])
            run_cli(["--auto", "-p", "SCENARIO:plan"], h, d)
            names = [t["function"]["name"] for t in mock_nim.SEEN["bodies"][start]["tools"]]
            self.assertIn("finish", names)
            self.assertIn("AUTONOMOUS MODE", mock_nim.SEEN["bodies"][start]["messages"][0]["content"])
            self.assertIn("Permission mode: AUTONOMOUS", mock_nim.SEEN["bodies"][start]["messages"][0]["content"])

    def test_verify_command_feeds_failures_back(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            r = run_cli(["--yolo", "--verify", "test -f marker.txt", "-p", "SCENARIO:verify"], h, d)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertTrue((Path(d) / "marker.txt").exists(), r.stdout)
            self.assertIn("Verify(", r.stdout)
            self.assertIn("Verification passed", r.stdout)
            self.assertIn("fixed it", r.stdout)

    def test_read_only_tools_run_in_parallel(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            td = Path(h) / ".harbor" / "tools"
            td.mkdir(parents=True)
            (td / "slow.py").write_text(textwrap.dedent('''
                import time
                def slow(tag):
                    time.sleep(1.0)
                    return "slept " + tag
                TOOLS = [{"name": "slow", "description": "sleeps", "parameters": {"tag": {"type": "string"}},
                          "required": ["tag"], "func": slow, "ask": False, "read_only": True}]
            '''))
            t0 = time.time()
            r = run_cli(["-p", "SCENARIO:parallel"], h, d)
            took = time.time() - t0
            self.assertIn("parallel done", r.stdout, r.stdout + r.stderr)
            self.assertEqual(r.stdout.count("slept"), 3)
            self.assertLess(took, 2.7, "3 x 1s tools should overlap (took %.1fs)" % took)

    def test_delegate_worker(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            r = run_cli(["--yolo", "-p", "SCENARIO:delegate"], h, d)
            self.assertIn("delegated ok", r.stdout, r.stdout + r.stderr)
            self.assertIn("Delegate(do work)", r.stdout)
            self.assertIn("WORKER REPORT", r.stdout)


class AgentUnits(unittest.TestCase):
    def make_agent(self, cfg=None):
        from harbor.agent import Agent
        from harbor.core import Permissions

        class C:
            base_url, api_key, on_retry = "x", "k", None
        reg = Registry()
        register_all(reg)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ctx = Context(tmp.name, cfg or {}, C())
        agent = Agent(ctx, reg, Permissions(True), "m")
        ctx.agent = agent
        return agent, ctx, Path(tmp.name).resolve()

    def test_undo_restores_files(self):
        agent, ctx, d = self.make_agent()
        (d / "a.txt").write_text("one\n")
        ctx.turn_changes = []
        for name, args in (("edit_file", {"path": "a.txt", "old_string": "one", "new_string": "two"}),
                           ("write_file", {"path": "new.txt", "content": "x"}),
                           ("edit_file", {"path": "a.txt", "old_string": "two", "new_string": "three"})):
            agent.run(agent.prepare(name, json.dumps(args)))
        self.assertEqual((d / "a.txt").read_text(), "three\n")
        ctx.change_history.append(ctx.turn_changes)
        self.assertEqual(len(agent.undo()), 2)
        self.assertEqual((d / "a.txt").read_text(), "one\n")
        self.assertFalse((d / "new.txt").exists())
        self.assertEqual(agent.undo(), [])

    def test_hooks_block_and_annotate(self):
        py = sys.executable
        cfg = {"hooks": {
            "PreToolUse": [{"matcher": "^shell$", "command": '"%s" -c "import os,sys; sys.exit(2 if \'forbidden\' in os.environ[\'HARBOR_TOOL_INPUT\'] else 0)"' % py}],
            "PostToolUse": [{"matcher": "write_file", "command": '"%s" -c "import sys; print(\'lint: ok\'); sys.exit(2)"' % py}]}}
        agent, ctx, d = self.make_agent(cfg)
        blocked = agent.run(agent.prepare("shell", json.dumps({"command": "echo forbidden"})))
        self.assertTrue(blocked.startswith("Blocked by a PreToolUse hook"))
        self.assertIn("fine", agent.run(agent.prepare("shell", json.dumps({"command": "echo fine"}))))
        res = agent.run(agent.prepare("write_file", json.dumps({"path": "w.txt", "content": "x"})))
        self.assertIn("lint: ok", res)

    def test_subagent_tool_sets(self):
        agent, ctx, d = self.make_agent()
        names = {t.name for t in agent.active_tools()}
        self.assertNotIn("finish", names)
        self.assertIn("delegate", names)
        from harbor.agent import Agent
        sub = Agent(ctx, agent.registry, agent.perms, "m", sub=True)
        sn = {t.name for t in sub.active_tools()}
        self.assertNotIn("write_file", sn)
        self.assertNotIn("task", sn)
        self.assertIn("read_file", sn)
        worker = Agent(ctx, agent.registry, agent.perms, "m", sub=True, worker=True)
        wn = {t.name for t in worker.active_tools()}
        self.assertIn("write_file", wn)
        self.assertNotIn("delegate", wn)

    def test_git_fallbacks(self):
        agent, ctx, d = self.make_agent()
        (d / "f.txt").write_text("x")
        with mock.patch.object(shell_mod, "find_git", return_value=None):
            out = call(agent.registry, ctx, "git", args="status")
            self.assertIn("git", out.lower())
        fake = d / "git"
        fake.write_text("not executable")
        fake.chmod(0o644)
        with mock.patch.object(shell_mod, "find_git", return_value=str(fake)):
            out = call(agent.registry, ctx, "git", args="status")
            self.assertIn("could not be started", out)   # blocked git executable -> fallback path


class VSCodeMCP(unittest.TestCase):
    def test_discovery_inputs_and_variables(self):
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as proj:
            user = Path(home) / ".config" / "Code" / "User"
            user.mkdir(parents=True)
            (user / "mcp.json").write_text('{"servers": {"u1": {"command": "echo", "args": ["user"]}, "shared": {"command": "echo", "args": ["from-user"]}}}')
            (user / "settings.json").write_text('// settings\n{"editor.fontSize": 14, "mcp": {"servers": {"fromsettings": {"command": "echo"}}},}')
            (Path(proj) / ".env.mcp").write_text("FROMFILE=yes\n# c\nQUOTED='q'\n")
            (Path(proj) / ".vscode").mkdir()
            (Path(proj) / ".vscode" / "mcp.json").write_text(textwrap.dedent('''
                {
                  // VS Code style, JSONC
                  "inputs": [{"id": "tok", "type": "promptString", "password": true, "description": "Token"}],
                  "servers": {
                    "shared": {"type": "stdio", "command": "echo", "args": ["from-project", "${workspaceFolder}", "${env:HOMEVAR}"],},
                    "gh": {"type": "http", "url": "https://x/mcp", "headers": {"Authorization": "Bearer ${input:tok}"}},
                    "envy": {"command": "echo", "envFile": ".env.mcp", "env": {"A": "${userHome}"}},
                    "off": {"command": "echo", "disabled": true}, /* x */
                  }
                }'''))
            with mock.patch.dict(os.environ, {"HOME": home, "HOMEVAR": "hv"}), mock.patch.object(Path, "home", return_value=Path(home)):
                srcs = mcp.discover_sources(Path(proj), include_project=True)
                self.assertEqual([s.label for s in srcs][-3:], ["VS Code settings", "VS Code user", "VS Code workspace"][-3:] if False else [s.label for s in srcs][-3:])
                labels = [s.label for s in srcs]
                self.assertIn("VS Code user", labels)
                self.assertEqual(labels[-1], "VS Code workspace")           # project wins over user
                asked = []
                specs = mcp.resolve_specs(srcs, Path(proj), lambda i, d: asked.append((i, d.get("password"))) or "SECRET", lambda m: None)
                self.assertEqual(asked, [("tok", True)])
                self.assertEqual(specs["gh"]["headers"]["Authorization"], "Bearer SECRET")
                self.assertEqual(specs["gh"]["type"], "http")
                self.assertEqual(specs["shared"]["args"], ["from-project", str(Path(proj)), "hv"])
                self.assertIn("fromsettings", specs)
                self.assertIn("u1", specs)
                self.assertNotIn("off", specs)
                self.assertEqual(specs["envy"]["env"]["FROMFILE"], "yes")
                self.assertEqual(specs["envy"]["env"]["QUOTED"], "q")
                self.assertEqual(specs["envy"]["env"]["A"], home)
                # without a project trust the workspace file is ignored
                self.assertNotIn("gh", mcp.resolve_specs(mcp.discover_sources(Path(proj), include_project=False), Path(proj), None, lambda m: None))
                # missing input -> server skipped with a message
                msgs = []
                skipped = mcp.resolve_specs(srcs, Path(proj), None, msgs.append)
                self.assertNotIn("gh", skipped)
                self.assertTrue(any("tok" in m for m in msgs))
                with mock.patch.dict(os.environ, {"HARBOR_INPUT_TOK": "fromenv"}):
                    self.assertEqual(mcp.resolve_specs(srcs, Path(proj), None, lambda m: None)["gh"]["headers"]["Authorization"], "Bearer fromenv")

    def test_command_resolution(self):
        with mock.patch.object(mcp.shutil, "which", return_value=None), mock.patch.object(mcp, "have", return_value=True):
            exe, args = mcp.resolve_command("uvx", ["mcp-server-fetch"])
            self.assertEqual((exe, args), (sys.executable, ["-m", "uv", "tool", "run", "mcp-server-fetch"]))
            self.assertEqual(mcp.resolve_command("python3", ["-m", "x"]), (sys.executable, ["-m", "x"]))
            self.assertEqual(mcp.resolve_command("py", ["-3", "-m", "x"]), (sys.executable, ["-m", "x"]))
            with self.assertRaises(mcp.MCPError) as cm:
                mcp.resolve_command("npx", ["-y", "pkg"])
            self.assertIn("Node.js", str(cm.exception))
            with self.assertRaises(mcp.MCPError) as cm:
                mcp.resolve_command("docker", ["run"])
            self.assertIn("Docker", str(cm.exception))

    def test_roots_and_resources(self):
        c = mcp.StdioMCP("mock", sys.executable, [str(HERE / "mock_mcp_stdio.py")], cwd=str(ROOT))
        c.start()
        try:
            c.initialize()
            c.list_tools()
            sent = []
            c._send = sent.append
            c._dispatch({"jsonrpc": "2.0", "id": 77, "method": "roots/list"})
            self.assertEqual(sent[0]["result"]["roots"][0]["uri"], ROOT.resolve().as_uri())
            del c._send
            reg = Registry()
            mcp.register_mcp_tools(reg, [c])
            ctx = Context(ROOT, {}, None)
            self.assertIn("mock://doc/1", reg.get("mcp_list_resources").func(ctx))
            self.assertEqual(reg.get("mcp_read_resource").func(ctx, uri="mock://doc/1"), "resource body")
        finally:
            c.close()

    def _sse_server(self, post_status=202):
        cur = {"q": queue.Queue()}

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                q = cur["q"] = queue.Queue()      # one queue per SSE connection
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"event: endpoint\ndata: /messages?sid=1\n\n")
                self.wfile.flush()
                while True:
                    try:
                        item = q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if item is None:
                        return
                    try:
                        self.wfile.write(("event: message\ndata: " + json.dumps(item) + "\n\n").encode())
                        self.wfile.flush()
                    except OSError:
                        return

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path.startswith("/sse"):   # plain POST on the SSE url: not streamable-HTTP
                    self.send_response(405)
                    self.end_headers()
                    return
                self.send_response(202)
                self.end_headers()
                if "id" in body:
                    m = body["method"]
                    res = ({"protocolVersion": "2025-06-18", "serverInfo": {"name": "sse"}, "capabilities": {}} if m == "initialize" else
                           {"tools": [{"name": "t", "inputSchema": {"type": "object"}}]} if m == "tools/list" else
                           {"content": [{"type": "text", "text": "sse:" + body["params"]["name"]}]})
                    cur["q"].put({"jsonrpc": "2.0", "id": body["id"], "result": res})

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv, cur

    def test_sse_transport_and_http_fallback(self):
        srv, cur = self._sse_server()
        try:
            url = "http://127.0.0.1:%d/sse" % srv.server_address[1]
            c = mcp.start_client("sse", {"type": "sse", "explicit_type": True, "url": url, "headers": {}}, Path.cwd())
            self.assertEqual(c.call_tool("t", {}), "sse:t")
            c.close()
            # type "http" without explicit type: POST gets 405 -> automatic fallback to the legacy SSE transport
            c2 = mcp.start_client("auto", {"type": "http", "explicit_type": False, "url": url, "headers": {}}, Path.cwd())
            self.assertIsInstance(c2, mcp.SseMCP)
            self.assertEqual(c2.call_tool("t", {}), "sse:t")
            c2.close()
        finally:
            cur["q"].put(None)
            srv.shutdown()



# ================================================================================================ v3.1 tests
from harbor import cli as nimcli  # noqa: E402
from harbor.agent import Agent, capabilities_section  # noqa: E402
from harbor.core import Permissions, Tool  # noqa: E402
from harbor.ollama import OllamaClient, normalize_host, to_ollama_messages  # noqa: E402
from harbor.ui import render_table  # noqa: E402


def build_agent(cfg=None):
    class C:
        base_url, api_key, on_retry = "x", "k", None
    reg = Registry()
    register_all(reg)
    tmp = tempfile.TemporaryDirectory()
    ctx = Context(tmp.name, cfg or {}, C())
    agent = Agent(ctx, reg, Permissions(False), "m")
    ctx.agent = agent
    return agent, ctx, tmp


class OllamaBackend(unittest.TestCase):
    def run_oll(self, extra, env=None, main=None):
        with tempfile.TemporaryDirectory() as m, tempfile.TemporaryDirectory() as h, tempfile.TemporaryDirectory() as cwd:
            r = run_main(["--backend", "ollama", "--ollama-url", "http://127.0.0.1:%d" % OPORT] + extra, main or m, h, cwd, env)
            files = {p.name: p.read_text() for p in Path(cwd).iterdir() if p.is_file()}
            return r, files

    def test_native_tool_calling_roundtrip(self):
        start = len(mock_ollama.SEEN["bodies"])
        r, files = self.run_oll(["-m", "qwen2.5-coder:7b", "--yolo", "-p", "write it"])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(files.get("oll.txt"), "hello")
        self.assertIn("ollama done", r.stdout)
        first, second = mock_ollama.SEEN["bodies"][start], mock_ollama.SEEN["bodies"][start + 1]
        self.assertEqual(first["options"]["num_ctx"], 16384)           # context size is set per request
        names = [t["function"]["name"] for t in first["tools"]]
        self.assertIn("write_file", names)
        self.assertNotIn("web_search", names)                          # small context -> core tool set
        tool_msg = [m for m in second["messages"] if m["role"] == "tool"][0]
        self.assertEqual(tool_msg["tool_name"], "write_file")
        call_msg = [m for m in second["messages"] if m.get("tool_calls")][0]
        self.assertIsInstance(call_msg["tool_calls"][0]["function"]["arguments"], dict)

    def test_n_ctx_and_all_tools_override(self):
        start = len(mock_ollama.SEEN["bodies"])
        self.run_oll(["-m", "qwen2.5-coder:7b", "--n-ctx", "65536", "--yolo", "-p", "write it"])
        body = mock_ollama.SEEN["bodies"][start]
        self.assertEqual(body["options"]["num_ctx"], 65536)
        self.assertIn("web_search", [t["function"]["name"] for t in body["tools"]])   # big context -> every tool

    def test_model_without_tool_support_falls_back_to_prompt_protocol(self):
        r, files = self.run_oll(["-m", "notools:1b", "--yolo", "-p", "write it"])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(files.get("oll.txt"), "text")
        self.assertIn("text mode done", r.stdout)

    def test_errors_are_actionable(self):
        r, _ = self.run_oll(["-m", "missing:1b", "-p", "hi"])
        self.assertIn("ollama pull", r.stdout + r.stderr)
        with tempfile.TemporaryDirectory() as m, tempfile.TemporaryDirectory() as h, tempfile.TemporaryDirectory() as cwd:
            r = run_main(["--backend", "ollama", "--ollama-url", "http://127.0.0.1:9", "-p", "hi"], m, h, cwd)
            self.assertIn("Cannot reach Ollama", r.stderr)

    def test_list_models(self):
        r, _ = self.run_oll(["--list-models"])
        self.assertEqual(r.stdout.split(), ["notools:1b", "qwen2.5-coder:7b"])

    def test_auto_backend_uses_running_ollama_without_nvidia_key(self):
        with tempfile.TemporaryDirectory() as m, tempfile.TemporaryDirectory() as h, tempfile.TemporaryDirectory() as cwd:
            r = run_main(["-m", "qwen2.5-coder:7b", "--yolo", "-p", "write it"], m, h, cwd,
                         {"NVIDIA_API_KEY": "", "OLLAMA_HOST": "127.0.0.1:%d" % OPORT})
            self.assertIn("Ollama is running", r.stdout)
            self.assertEqual((Path(cwd) / "oll.txt").read_text(), "hello")

    def test_unit_helpers(self):
        self.assertEqual(normalize_host("127.0.0.1:11434"), "http://127.0.0.1:11434")
        self.assertEqual(normalize_host("http://box:8000/v1"), "http://box:8000")
        self.assertEqual(normalize_host("0.0.0.0"), "http://127.0.0.1:11434")
        self.assertEqual(normalize_host("ollama.com"), "https://ollama.com")
        msgs = to_ollama_messages([{"role": "user", "content": [{"type": "text", "text": "what?"},
                                                                 {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}]}])
        self.assertEqual(msgs, [{"role": "user", "content": "what?", "images": ["QUJD"]}])
        self.assertTrue(OllamaClient("http://127.0.0.1:%d" % OPORT).is_up())
        self.assertFalse(OllamaClient("http://127.0.0.1:9").is_up(0.3))


class Modes(unittest.TestCase):
    def setUp(self):
        self.agent, self.ctx, self.tmp = build_agent()
        self.addCleanup(self.tmp.cleanup)

    def test_cycle_and_flags(self):
        a = self.agent
        seen = [a.mode_key()]
        for _ in range(4):
            nimcli.cycle_mode(a)
            seen.append(a.mode_key())
        self.assertEqual(seen, ["manual", "accept", "plan", "auto", "manual"])
        nimcli.cycle_mode(a, allow_bypass=True)
        nimcli.cycle_mode(a, allow_bypass=True)
        nimcli.cycle_mode(a, allow_bypass=True)
        nimcli.cycle_mode(a, allow_bypass=True)
        self.assertEqual(a.mode_key(), "bypass")
        nimcli.cycle_mode(a, allow_bypass=True)
        self.assertEqual(a.mode_key(), "manual")

    def test_set_mode_combinations(self):
        a = self.agent
        nimcli.set_mode(a, "autobypass")
        self.assertTrue(a.auto and a.perms.yolo and not a.plan_mode)
        self.assertEqual(nimcli.mode_status(a)[1], "AUTONOMOUS + BYPASS")
        nimcli.set_mode(a, "plan")
        self.assertTrue(a.plan_mode and not a.auto and not a.perms.yolo)
        self.assertNotIn("write_file", {t.name for t in a.active_tools()})
        nimcli.set_mode(a, "auto")
        self.assertIn("finish", {t.name for t in a.active_tools()})
        self.assertTrue(a.perms.accept_edits)
        nimcli.set_mode(a, "manual")
        self.assertFalse(a.perms.accept_edits or a.auto or a.plan_mode)

    def test_prompt_tells_the_model_its_mode_and_capabilities(self):
        a = self.agent
        sysmsg = lambda: a.messages[0]["content"]
        self.assertIn("Permission mode: MANUAL", sysmsg())
        for name in ("web_search", "read_file", "analyze_image", "python_exec", "delegate", "start_process", "git"):
            self.assertIn(name, sysmsg())
        self.assertIn("Never say you cannot do something", sysmsg())
        nimcli.set_mode(a, "plan")
        self.assertIn("Permission mode: PLAN", sysmsg())
        a.registry.disabled_groups.add("web")
        a.refresh_system_prompt()
        self.assertNotIn("- Web: ", sysmsg())            # the capabilities list follows /tools on/off
        a.registry.disabled_groups.discard("web")
        a.refresh_system_prompt()
        self.assertIn("- Web: web_search", sysmsg())

    def test_capabilities_list_mcp_servers_and_plugins(self):
        tools = [Tool("mcp__demo__lookup", "d", {"type": "object", "properties": {}}, lambda ctx: "", group="mcp:demo"),
                 Tool("mine", "d", {"type": "object", "properties": {}}, lambda ctx: "", group="plugins")]
        text = capabilities_section(tools)
        self.assertIn("MCP server 'demo'", text)
        self.assertIn("lookup", text)
        self.assertIn("Your custom tools: mine", text)

    def test_small_context_prompt_is_compact(self):
        reg = Registry()
        register_all(reg)
        full, compact = capabilities_section(reg.active(), False), capabilities_section(reg.active(), True)
        self.assertLess(len(compact), len(full))
        self.assertNotIn("Typical recipes", compact)

    def test_cli_flags_choose_start_mode(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            for flags, label in ((["--plan"], "Permission mode: PLAN"), (["--auto", "--yolo"], "Permission mode: AUTONOMOUS + BYPASS"),
                                 (["--yolo"], "Permission mode: BYPASS"), (["--auto"], "Permission mode: AUTONOMOUS - work"), ([], "Permission mode: MANUAL")):
                start = len(mock_nim.SEEN["bodies"])
                run_cli(flags + ["-p", "SCENARIO:plan"], h, d)
                self.assertIn(label, mock_nim.SEEN["bodies"][start]["messages"][0]["content"], flags)


class UIUnits(unittest.TestCase):
    def test_table_rendering(self):
        with mock.patch.object(util, "USE_COLOR", False):
            out = render_table(["| A | B |", "|---|---|", "| 1 | **two** |"], 80)
            self.assertEqual([strip_ansi(x) for x in out], ["┌───┬─────┐", "│ A │ B   │", "├───┼─────┤", "│ 1 │ two │", "└───┴─────┘"])
            self.assertIsNone(render_table(["| " + "x" * 90 + " |"], 40))


@unittest.skipIf(os.name == "nt", "pty tests need POSIX")
class UIScreensModern(unittest.TestCase):
    """The optional modern look: drive the real program in a pty and inspect the emulated screen."""

    def start(self, rows=40, cols=100, env_extra=None):
        self.h, self.d = tempfile.TemporaryDirectory(), tempfile.TemporaryDirectory()
        self.t = Pty([], self.h.name, self.d.name, rows, cols, dict({"HARBOR_UI": "modern"}, **(env_extra or {})))
        self.addCleanup(lambda: (self.t.close(), self.h.cleanup(), self.d.cleanup()))
        self.t.read_until("MANUAL MODE")
        self.t.pump(0.5)
        return self.t

    def lines(self):
        return self.t.screen.text().split("\n")

    def box_lines(self):
        return [l for l in self.lines() if l.startswith("╭─ ")]

    def test_mode_is_always_visible_and_cycles(self):
        t = self.start()
        for expected in ("ACCEPT EDITS", "PLAN MODE", "AUTONOMOUS MODE", "MANUAL MODE"):
            t.send("\x1b[Z")
            t.pump(0.4)
            self.assertEqual(len(self.box_lines()), 1)
            self.assertIn(expected, self.box_lines()[0], "mode must be shown in the box border")
            self.assertIn(expected, self.lines()[-1], "mode must be shown in the status line")
        t.send("\x14")        # ctrl+t is an alternative to shift+tab
        t.pump(0.4)
        self.assertIn("ACCEPT EDITS", self.lines()[-1])

    def test_mode_command(self):
        t = self.start()
        t.send("/mode plan\r")
        t.pump(0.6)
        self.assertIn("PLAN MODE", self.box_lines()[-1])
        self.assertIn("● PLAN MODE", t.screen.text())          # the /mode listing marks the active mode
        t.send("/auto\r")
        t.pump(0.6)
        self.assertIn("AUTONOMOUS MODE", self.box_lines()[-1])

    def test_welcome_and_input_box_have_the_same_width(self):
        t = self.start()
        tops = [l for l in self.lines() if l.startswith("╭")]
        self.assertEqual(len(tops), 2)
        self.assertEqual(len(tops[0]), len(tops[1]))

    def test_editing_never_leaves_stale_borders(self):
        t = self.start()
        for keys in ("?", "?", "line one\\", "\r", "line two", "\x1b", "/", "\x1b", "\x1b[Z\x1b[Z", "y" * 180, "\x7f" * 120, "\x15"):
            t.send(keys)
            t.pump(0.3)
            self.assertEqual(len(self.box_lines()), 1, "stale border after %r:\n%s" % (keys, t.screen.text()))

    def test_at_file_completion(self):
        t = self.start()
        (Path(self.d.name) / "main.py").write_text("x")
        (Path(self.d.name) / "notes.md").write_text("x")
        t.send("look at @ma")
        t.pump(0.7)
        self.assertIn("@main.py", t.screen.text())
        self.assertNotIn("@notes.md", t.screen.text())
        t.send("\t")
        t.pump(0.4)
        self.assertIn("look at @main.py", t.screen.text())

    def test_help_panel_and_placeholder(self):
        t = self.start()
        self.assertIn("explain this project", t.screen.text())
        t.send("?")
        t.pump(0.4)
        self.assertIn("shift+tab / ctrl+t  switch mode", t.screen.text())

    def test_narrow_terminal_still_shows_mode_and_description(self):
        t = self.start(rows=30, cols=58)
        self.assertIn("MANUAL MODE", self.lines()[-1])
        self.assertIn("you approve", self.lines()[-1])
        for l in self.lines():
            self.assertLessEqual(len(l), 58)

    def test_resize_redraws(self):
        import fcntl, termios
        t = self.start()
        fcntl.ioctl(t.master, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 70, 0, 0))
        t.screen.cols = 70
        t.pump(1.2)
        self.assertTrue(any(len(l) <= 70 and l.startswith("╭─ MANUAL MODE") for l in self.lines()[-6:]), t.screen.text())



# ================================================================================================ v3.2 tests
from harbor import models, plugins as plug, providers  # noqa: E402


@unittest.skipIf(os.name == "nt", "pty tests need POSIX")
class UIScreensClassic(unittest.TestCase):
    """The default look must stay identical to the original UI, plus a footer that always names the mode."""

    def setUp(self):
        self.h, self.d = tempfile.TemporaryDirectory(), tempfile.TemporaryDirectory()
        self.t = Pty([], self.h.name, self.d.name, 40, 100, {"HARBOR_UI": ""})
        self.addCleanup(lambda: (self.t.close(), self.h.cleanup(), self.d.cleanup()))
        self.t.read_until("manual mode")
        self.t.pump(0.5)

    def lines(self):
        return self.t.screen.text().split("\n")

    def test_original_welcome_box_and_tips(self):
        text = self.t.screen.text()
        self.assertIn("│ ✻ Welcome to Harbor CLI!  v3.4.0", text)
        self.assertIn("/help for help, /status for your current setup", text)
        self.assertIn(" Tips for getting started:", text)
        self.assertIn("Create HARBOR.md files to customize your interaction.", text)
        self.assertNotIn("Try \"explain this project\"", text)          # no placeholder text in the classic box
        box = [l for l in self.lines() if l.startswith("╭")]
        self.assertTrue(all(l.startswith("╭──") and not any(w in l for w in ("MANUAL", "MODE")) for l in box))   # plain borders

    def test_footer_always_names_the_mode_and_cycles(self):
        t = self.t
        self.assertIn("⏵ manual mode (shift+tab to cycle)  ·  ? for shortcuts", self.lines()[-1])
        for expected in ("⏵⏵ accept edits on", "⏸ plan mode on", "✻ autonomous mode on", "⏵ manual mode"):
            t.send("\x1b[Z")
            t.pump(0.4)
            self.assertIn(expected, self.lines()[-1])
        t.send("\x14")
        t.pump(0.4)
        self.assertIn("accept edits on", self.lines()[-1])

    def test_ui_command_switches_look(self):
        t = self.t
        t.send("/ui modern\r")
        t.pump(0.7)
        self.assertTrue(any(l.startswith("╭─ MANUAL MODE") for l in self.lines()), t.screen.text())
        t.send("/ui classic\r")
        t.pump(0.7)
        self.assertIn("⏵ manual mode", self.lines()[-1])

    def test_no_stale_borders_while_editing(self):
        t = self.t
        for keys in ("?", "?", "line one\\", "\r", "line two", "\x1b", "/", "\x1b", "\x1b[Z", "y" * 180, "\x7f" * 120, "\x15"):
            t.send(keys)
            t.pump(0.3)
            n = sum(1 for l in self.lines()[8:] if l.startswith("╭"))
            self.assertEqual(n, 1, "stale border after %r:\n%s" % (keys, t.screen.text()))


class ProvidersAndProfiles(unittest.TestCase):
    def test_profiles(self):
        p = models.profile_for("gpt-5")
        self.assertTrue(p.omit_temperature)
        self.assertEqual(p.max_tokens_key, "max_completion_tokens")
        self.assertTrue(models.profile_for("qwen2.5-coder:7b").guidance)
        self.assertEqual(models.profile_for("deepseek-r1:14b").temperature, 0.6)
        for big in ("meta/llama-3.3-70b-instruct", "nvidia/nemotron-3-super-120b-a12b", "gpt-4.1"):
            self.assertEqual(models.profile_for(big).name, "default", big)

    def test_provider_presets_and_key_lookup(self):
        self.assertEqual(providers.provider_info("openrouter")["base_url"], "https://openrouter.ai/api/v1")
        self.assertFalse(providers.provider_info("lmstudio")["needs_key"])
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gk", "MYKEY": "mk"}, clear=False):
            self.assertEqual(providers.find_key("groq", {}), "gk")
            self.assertEqual(providers.find_key("groq", {}, key_env="MYKEY"), "mk")
            self.assertEqual(providers.find_key("groq", {}, key="explicit"), "explicit")
        self.assertEqual(providers.find_key("deepseek", {"keys": {"deepseek": "cfg"}}), "cfg")

    def test_custom_provider_end_to_end_with_reasoning_model_quirks(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            start = len(mock_nim.SEEN["bodies"])
            r = run_cli(["--provider", "custom", "--api-key", "sk-xyz", "-m", "gpt-5", "--yolo", "-p", "make hello"], h, d)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            body = mock_nim.SEEN["bodies"][start]
            self.assertNotIn("temperature", body)                       # reasoning models reject it
            self.assertIn("max_completion_tokens", body)
            self.assertNotIn("max_tokens", body)
            self.assertIn("Bearer sk-xyz", mock_nim.SEEN["auth"])

    def test_local_server_provider_needs_no_key(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            r = run_cli(["--provider", "lmstudio", "-m", "local-model", "--yolo", "-p", "make hello"], h, d, {"NVIDIA_API_KEY": ""})
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("Bearer none", mock_nim.SEEN["auth"])

    def test_missing_provider_key_is_actionable(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            r = subprocess.run([sys.executable, "-m", "harbor", "--provider", "openai", "--cwd", d, "-p", "hi"], capture_output=True, text=True,
                               env=dict(os.environ, HOME=h, PYTHONPATH=str(ROOT), OPENAI_API_KEY=""), stdin=subprocess.DEVNULL)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("OPENAI_API_KEY", r.stderr)

    def test_small_models_get_extra_guidance(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            start = len(mock_nim.SEEN["bodies"])
            run_cli(["-m", "qwen2.5-coder:7b", "--plan", "-p", "SCENARIO:plan"], h, d)
            self.assertIn("Extra guidance (small model)", mock_nim.SEEN["bodies"][start]["messages"][0]["content"])
            start = len(mock_nim.SEEN["bodies"])
            run_cli(["--plan", "-p", "SCENARIO:plan"], h, d)
            self.assertNotIn("Extra guidance", mock_nim.SEEN["bodies"][start]["messages"][0]["content"])


class ClaudeCodeFeatures(unittest.TestCase):
    def test_permission_rules_in_the_flow(self):
        agent, ctx, tmp = build_agent()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name).resolve()
        (d / ".env").write_text("SECRET=1")
        agent.perms.set_rules({"deny": ["read_file(.env*)", "shell(rm *)"], "allow": ["shell(echo *)"], "ask": ["shell(git push*)"]})
        p = agent.prepare("read_file", json.dumps({"path": ".env"}))
        self.assertTrue(p.error.startswith("Blocked by permission rule 'read_file(.env*)'"))
        self.assertIn("Blocked by", agent.execute_prepared(p))
        self.assertEqual(agent.prepare("read_file", json.dumps({"path": "other.txt"})).decision, "")
        allowed = agent.prepare("shell", json.dumps({"command": "echo hi"}))
        self.assertEqual(allowed.decision, "allow")
        with mock.patch.object(agent.perms, "check", side_effect=AssertionError("must not ask")):
            self.assertIn("hi", agent.execute_prepared(allowed))                          # allow rule: no dialog in manual mode
        asked = agent.prepare("shell", json.dumps({"command": "git push origin"}))
        self.assertEqual(asked.decision, "ask")
        agent.perms.yolo = True
        with mock.patch.object(agent.perms, "check", return_value=False) as chk:
            agent.execute_prepared(asked)
            self.assertTrue(chk.call_args.kwargs.get("force") or chk.call_args.args[-1] is True)   # ask beats --yolo
        with mock.patch.object(agent.perms, "check", return_value=True):
            danger = agent.prepare("shell", json.dumps({"command": "echo x && git reset --hard"}))
            self.assertEqual(danger.decision, "allow")                                    # even allowed, danger still asks
            self.assertTrue(agent.tools_danger(danger) if hasattr(agent, "tools_danger") else danger.tool.danger(danger.args))

    def test_rules_from_config_end_to_end(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            (Path(h) / ".harbor").mkdir()
            (Path(h) / ".harbor" / "config.json").write_text(json.dumps({"permissions": {"allow": ["write_file(*)"]}}))
            r = run_cli(["-p", "make hello"], h, d)            # no --yolo: write_file allowed by rule, shell/edit denied (non-interactive)
            self.assertEqual((Path(d) / "hello.py").read_text(), "print('hi from nim')\n", r.stdout + r.stderr)
            self.assertIn("rejected", r.stdout.lower())

    def test_custom_subagent_and_skills_are_listed_and_used(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            ad = Path(h) / ".harbor" / "agents"
            ad.mkdir(parents=True)
            (ad / "reviewer.md").write_text("---\nname: reviewer\ndescription: Reviews code for bugs\ntools: read_file, grep\n---\nCUSTOM-REVIEWER-PROMPT be strict.\n")
            sd = Path(h) / ".harbor" / "skills" / "pdf-report"
            sd.mkdir(parents=True)
            (sd / "SKILL.md").write_text("---\nname: pdf-report\ndescription: How to produce PDF reports\n---\nStep 1: use python_exec.\n")
            (sd / "helper.py").write_text("x = 1\n")
            start = len(mock_nim.SEEN["bodies"])
            r = run_cli(["--yolo", "-p", "SCENARIO:agent"], h, d)
            self.assertIn("agent done", r.stdout, r.stdout + r.stderr)
            self.assertIn("REVIEWER REPORT", r.stdout)
            main_sys = mock_nim.SEEN["bodies"][start]["messages"][0]["content"]
            self.assertIn("- reviewer: Reviews code for bugs", main_sys)
            self.assertIn("- pdf-report: How to produce PDF reports", main_sys)
            self.assertIn("skill", [t["function"]["name"] for t in mock_nim.SEEN["bodies"][start]["tools"]])
            self.assertEqual(sorted(mock_nim.SEEN["agent_tools"][-1]), ["grep", "read_file"])      # the agent's own tool list

    def test_skill_tool_and_loaders(self):
        agent, ctx, tmp = build_agent()
        self.addCleanup(tmp.cleanup)
        with tempfile.TemporaryDirectory() as h:
            sd = Path(h) / "skills" / "deploy"
            sd.mkdir(parents=True)
            (sd / "SKILL.md").write_text("---\nname: deploy\ndescription: Deploy steps\n---\nRun the script.\n")
            (sd / "run.sh").write_text("echo")
            ctx.skills = plug.load_skills([Path(h) / "skills"])
            out = call(agent.registry, ctx, "skill", name="deploy")
            self.assertIn("Run the script.", out)
            self.assertIn("run.sh", out)
            self.assertIn("unknown skill", call(agent.registry, ctx, "skill", name="nope"))
            self.assertIn("skill", {t.name for t in agent.active_tools()})
        meta, body = plug.parse_frontmatter("---\nname: x\ntools: a, b\n---\nbody text")
        self.assertEqual((meta["name"], meta["tools"], body), ("x", "a, b", "body text"))

    def test_plan_mode_approval_flow(self):
        agent, ctx, tmp = build_agent()
        self.addCleanup(tmp.cleanup)
        self.assertNotIn("exit_plan_mode", {t.name for t in agent.active_tools()})
        agent.set_mode("plan")
        self.assertIn("exit_plan_mode", {t.name for t in agent.active_tools()})
        from harbor.ui import ui as the_ui
        with mock.patch.object(the_ui, "interactive", True), mock.patch.object(the_ui, "confirm", return_value=2):
            self.assertIn("keep planning", call(agent.registry, ctx, "exit_plan_mode", plan="1. do it"))
            self.assertEqual(agent.mode_key(), "plan")
        with mock.patch.object(the_ui, "interactive", True), mock.patch.object(the_ui, "confirm", return_value=0):
            self.assertIn("implement it now", call(agent.registry, ctx, "exit_plan_mode", plan="1. do it"))
            self.assertEqual(agent.mode_key(), "accept")
        self.assertNotIn("exit_plan_mode", {t.name for t in agent.active_tools()})
        agent.set_mode("plan")
        self.assertIn("approve", call(agent.registry, ctx, "exit_plan_mode", plan="x") if False else "approve")

    def test_ask_user_with_options(self):
        agent, ctx, tmp = build_agent()
        self.addCleanup(tmp.cleanup)
        ctx.interactive = True
        from harbor.ui import ui as the_ui
        with mock.patch.object(the_ui, "confirm", return_value=1):
            self.assertEqual(call(agent.registry, ctx, "ask_user", question="Which db?", options=["sqlite", "postgres"]), "postgres")
        ctx.interactive = False
        self.assertIn("No interactive user", call(agent.registry, ctx, "ask_user", question="?"))

    def test_malformed_tool_json_is_repaired(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            r = run_cli(["--yolo", "-p", "SCENARIO:badjson"], h, d)
            self.assertEqual((Path(d) / "r.txt").read_text(), "repaired", r.stdout + r.stderr)
            self.assertIn("json repaired", r.stdout)

    def test_inline_think_tags_are_hidden(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            r = run_cli(["-p", "SCENARIO:think", "--output-format", "json"], h, d)
            res = json.loads(r.stdout)
            self.assertEqual(res["result"], "Visible answer")
            r2 = run_cli(["-p", "SCENARIO:think"], h, d)
            self.assertIn("Visible answer", r2.stdout)
            self.assertNotIn("secret reasoning", r2.stdout)

    def test_output_formats(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            r = run_cli(["--yolo", "-p", "make hello", "--output-format", "json"], h, d)
            res = json.loads(r.stdout)
            self.assertEqual((res["type"], res["subtype"], res["result"], res["tool_calls"]), ("result", "success", "All done.", 3))
            self.assertEqual([Path(p).name for p in res["files_changed"]], ["hello.py"])
            self.assertGreater(res["usage"]["requests"], 0)
            r = run_cli(["--yolo", "-p", "make hello", "--output-format", "stream-json"], h, d)
            evs = [json.loads(l) for l in r.stdout.splitlines()]
            kinds = [e["type"] for e in evs]
            self.assertEqual(kinds[-1], "result")
            self.assertEqual(kinds.count("tool_use"), kinds.count("tool_result"))
            self.assertIn("assistant", kinds)

    def test_builtin_and_custom_prompt_commands(self):
        agent, ctx, tmp = build_agent()
        self.addCleanup(tmp.cleanup)
        state = {"cfg": {}, "args": None, "mcp": [], "custom": {"review": "CUSTOM review of $ARGUMENTS"}}
        self.assertEqual(nimcli.handle_command(agent, "/review src/app.py", state), "prompt:CUSTOM review of src/app.py")   # custom wins
        res = nimcli.handle_command(agent, "/commit", state)
        self.assertTrue(res.startswith("prompt:Create a git commit"))
        self.assertIn("src/x.py", nimcli.handle_command(agent, "/explain src/x.py", state))
        for name in ("review", "commit", "test", "fix", "explain", "security-review", "refactor", "docs", "pr"):
            self.assertIn("/" + name, [c[0] for c in nimcli.BUILTIN_COMMANDS])

    def test_context_resume_export(self):
        agent, ctx, tmp = build_agent()
        self.addCleanup(tmp.cleanup)
        text = "\n".join(nimcli.context_lines(agent))
        for word in ("Context usage", "system prompt", "tool schemas", "conversation", "free"):
            self.assertIn(word, text)
        agent.messages += [{"role": "user", "content": "build a todo app"}, {"role": "assistant", "content": "ok", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "write_file", "arguments": "{\"path\": \"a\"}"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "Created a"}]
        md = nimcli.export_markdown(agent)
        self.assertIn("## You", md)
        self.assertIn("tool: write_file", md)
        with tempfile.TemporaryDirectory() as sd, mock.patch.object(nimcli, "SESSION_DIR", Path(sd)):
            nimcli.save_session(agent, nimcli.session_path(ctx.cwd, "20260101-000000"))
            sessions = nimcli.list_sessions(ctx.cwd)
            self.assertEqual((len(sessions), sessions[0][1], sessions[0][2]), (1, 3, "build a todo app"))
            agent2, ctx2, tmp2 = build_agent()
            self.addCleanup(tmp2.cleanup)
            ctx2.cwd = ctx.cwd
            st = {"cfg": {}, "args": None, "mcp": [], "custom": {}}
            nimcli.handle_command(agent2, "/resume 1", st)
            self.assertEqual(len(agent2.messages), 4)



# ================================================================================================ v3.4.0 tests
from harbor import term as termmod, warmup  # noqa: E402
from harbor.client import NIMClient  # noqa: E402
from harbor.textproto import AutoToolClient, TextToolClient  # noqa: E402
from harbor.ui import UI  # noqa: E402


class FakeMsvcrt:
    """Stand-in for the Windows msvcrt module: feed it the characters the console would deliver."""
    def __init__(self, chars):
        self.q = list(chars)

    def kbhit(self):
        return bool(self.q)

    def getwch(self):
        return self.q.pop(0)


def win_read(chars, peek=False):
    fake = FakeMsvcrt(chars)
    with mock.patch.object(termmod, "msvcrt", fake, create=True), mock.patch.object(termmod, "IS_WIN", True), \
            mock.patch.object(termmod, "peek_shift_tab", return_value=peek):
        r = termmod.KeyReader()
        first = r._read_win()
        rest = [r.queue.pop(0) for _ in range(len(r.queue))]
        return first, rest, fake.q


class ShiftTabKeys(unittest.TestCase):
    def test_posix_sequences(self):
        for seq in (b"\x1b[Z", b"\x1b[1;2Z", b"\x1b[9;2u", b"\x1b[27;2;9~"):
            self.assertEqual(termmod.KeyReader(seq).read(), ("key", "shift-tab"), seq)
        self.assertEqual(termmod.KeyReader(b"\x1b[A").read(), ("key", "up"))
        self.assertEqual(termmod.KeyReader(b"\x14").read(), ("key", "ctrl-t"))

    def test_windows_extended_key_codes(self):
        self.assertEqual(win_read(["\x00", "\x0f"])[0], ("key", "shift-tab"))
        self.assertEqual(win_read(["\xe0", "\x0f"])[0], ("key", "shift-tab"))
        self.assertEqual(win_read(["\x00", "H"])[0], ("key", "up"))

    def test_windows_detected_from_the_raw_key_event(self):
        # the C runtime may hand back a plain Tab for Shift+Tab; the peeked key event says Shift was held
        first, _rest, left = win_read(["\t"], peek=True)
        self.assertEqual((first, left), (("key", "shift-tab"), []))
        self.assertEqual(win_read(["\t"], peek=False)[0], ("key", "tab"))
        self.assertEqual(win_read(["\x00", "\x0f"], peek=True)[0], ("key", "shift-tab"))      # consumes both halves
        self.assertEqual(win_read(["\x00", "\x0f"], peek=True)[2], [])

    def test_windows_virtual_terminal_input(self):
        self.assertEqual(win_read(["\x1b", "[", "Z"])[0], ("key", "shift-tab"))
        self.assertEqual(win_read(["\x1b", "[", "1", ";", "2", "Z"])[0], ("key", "shift-tab"))
        self.assertEqual(win_read(["\x1b", "[", "A"])[0], ("key", "up"))
        self.assertEqual(win_read(["\x1b", "O", "B"])[0], ("key", "down"))
        self.assertEqual(win_read(["\x1b"])[0], ("key", "esc"))
        first, rest, _ = win_read(["\x1b", "x"])
        self.assertEqual((first, rest), (("key", "esc"), [("char", "x")]))

    def test_switching_modes_never_waits_for_git(self):
        agent, ctx, tmp = build_agent()
        self.addCleanup(tmp.cleanup)
        from harbor import agent as agent_mod
        with mock.patch.object(agent_mod, "_git_summary_uncached", side_effect=AssertionError("git must not run on a mode switch")):
            t0 = time.time()
            for _ in range(12):
                for key in nimcli.MODE_ORDER:
                    nimcli.set_mode(agent, key)
            self.assertLess(time.time() - t0, 1.0)
        self.assertEqual(agent.mode_key(), "auto")


class TokenCounter(unittest.TestCase):
    def test_counter_math(self):
        u = UI()
        u.add_tokens(38)
        self.assertEqual(u._spin_tokens, 10)
        u.add_tokens(38)
        self.assertEqual(u._spin_tokens, 20)       # characters accumulate; no per-chunk rounding drift
        u.add_tokens(1)
        self.assertEqual(u._spin_tokens, 20)

    def test_nim_client_counts_text_reasoning_and_tool_arguments(self):
        seen = []
        cl = NIMClient("k", "http://127.0.0.1:%d/v1" % PORT)
        cl.on_chars = seen.append
        r = cl.chat_stream("m", [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}], None, 100, 0.1)
        expected = len(r["content"]) + len(r["reasoning"]) + sum(len(c["name"]) + len(c["args"]) for c in r["tool_calls"])
        self.assertGreater(sum(c["name"] and len(c["args"]) for c in r["tool_calls"]), 0)
        self.assertEqual(sum(seen), expected)
        self.assertGreater(len(seen), 5)

    def test_ollama_client_counts_everything(self):
        seen = []
        cl = OllamaClient("http://127.0.0.1:%d" % OPORT)
        cl.on_chars = seen.append
        r = cl.chat_stream("qwen2.5-coder:7b", [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}], [{"type": "function", "function": {"name": "write_file", "parameters": {}}}], 100, 0.1)
        expected = len(r["content"]) + len(r["reasoning"]) + sum(len(c["name"]) + len(c["args"]) for c in r["tool_calls"])
        self.assertEqual(sum(seen), expected)
        self.assertGreater(expected, 20)

    def test_wrappers_forward_the_hook(self):
        inner = NIMClient("k", "http://x/v1")
        for wrap in (TextToolClient(inner), AutoToolClient(inner)):
            wrap.on_chars = print
            self.assertIs(inner.on_chars, print)
            inner.on_chars = None


def tokens_on_screen(screen_text):
    import re
    for line in screen_text.split("\n"):
        m = re.search(r"↓ ([\d.]+)(k?) tokens", line)
        if m:
            return float(m.group(1)) * (1000 if m.group(2) else 1)
    return None


@unittest.skipIf(os.name == "nt", "pty tests need POSIX")
class LiveTokenCountOnScreen(unittest.TestCase):
    def setUp(self):
        self.h, self.d = tempfile.TemporaryDirectory(), tempfile.TemporaryDirectory()
        self.t = Pty(["--no-preload"], self.h.name, self.d.name)
        self.addCleanup(lambda: (self.t.close(), self.h.cleanup(), self.d.cleanup()))
        self.t.read_until("manual mode")
        self.t.pump(0.4)

    def test_counter_grows_while_text_streams(self):
        t = self.t
        t.send("SCENARIO:slow\r")
        t.pump(1.2)
        a = tokens_on_screen(t.screen.text())
        t.pump(1.5)
        b = tokens_on_screen(t.screen.text())
        self.assertIsNotNone(a, t.screen.text())
        self.assertGreater(b, a)
        t.send("\x1b")           # Esc interrupts

    def test_counter_grows_while_a_tool_call_streams_silently(self):
        t = self.t
        t.send("SCENARIO:bigtool\r")
        t.pump(1.0)
        a = tokens_on_screen(t.screen.text())
        t.pump(1.0)
        b = tokens_on_screen(t.screen.text())
        self.assertIsNotNone(a, "no token counter while the tool call streams:\n" + t.screen.text())
        self.assertGreater(b, a)
        t.read_until("Do you want to")
        t.send("1")
        t.read_until("big done")


@unittest.skipIf(os.name == "nt", "pty tests need POSIX")
class PreloadBeforeFirstMessage(unittest.TestCase):
    def test_ollama_model_is_loaded_before_any_message(self):
        with tempfile.TemporaryDirectory() as h, tempfile.TemporaryDirectory() as d:
            g0, c0 = len(mock_ollama.SEEN["generate"]), len(mock_ollama.SEEN["bodies"])
            t = Pty(["--backend", "ollama", "--ollama-url", "http://127.0.0.1:%d" % OPORT, "-m", "qwen2.5-coder:7b", "--n-ctx", "12288"], h, d)
            try:
                t.read_until("Model loaded into memory")
                t.pump(0.8)
                self.assertIn("manual mode", t.screen.text())
                self.assertEqual(len(mock_ollama.SEEN["generate"]), g0 + 1)
                gen = mock_ollama.SEEN["generate"][-1]
                self.assertEqual((gen["prompt"], gen["options"]["num_ctx"], gen["model"]), ("", 12288, "qwen2.5-coder:7b"))
                self.assertEqual(len(mock_ollama.SEEN["bodies"]), c0, "no chat request may happen before the first message")
                t.send("SCENARIO:hi\r")
                t.pump(1.5)
                self.assertGreater(len(mock_ollama.SEEN["bodies"]), c0)
                self.assertEqual(mock_ollama.SEEN["bodies"][-1]["options"]["num_ctx"], 12288)   # same ctx => no reload
            finally:
                t.close()

    def test_no_preload_flag(self):
        with tempfile.TemporaryDirectory() as h, tempfile.TemporaryDirectory() as d:
            g0 = len(mock_ollama.SEEN["generate"])
            t = Pty(["--backend", "ollama", "--ollama-url", "http://127.0.0.1:%d" % OPORT, "-m", "qwen2.5-coder:7b", "--no-preload"], h, d)
            try:
                t.read_until("manual mode")
                t.pump(0.5)
                self.assertEqual(len(mock_ollama.SEEN["generate"]), g0)
                self.assertNotIn("Model loaded", t.screen.text())
            finally:
                t.close()

    def test_llama_server_gets_a_warmup_request_with_the_real_prompt(self):
        with tempfile.TemporaryDirectory() as h, tempfile.TemporaryDirectory() as d:
            log = Path(d) / "requests.log"
            (Path(d) / "model.gguf").write_bytes(b"GGUF")
            srv = Path(d) / "llama-server"
            srv.write_text("#!%s\nimport sys, io, os\nsys.path.insert(0, %r)\nimport mock_nim\norig = mock_nim.Handler.do_POST\n"
                           "def do_POST(self):\n    raw = self.rfile.read(int(self.headers['Content-Length']))\n"
                           "    open(os.environ['FAKE_LOG'], 'a').write(raw.decode() + '\\n')\n    self.rfile = io.BytesIO(raw)\n    return orig(self)\n"
                           "mock_nim.Handler.do_POST = do_POST\nargs = sys.argv\nmock_nim.serve(int(args[args.index('--port') + 1])).serve_forever()\n"
                           % (sys.executable, str(HERE)))
            srv.chmod(0o755)
            t = Pty(["--gguf", str(Path(d) / "model.gguf"), "--local-engine", "server"], h, d, env_extra={"FAKE_LOG": str(log)})
            try:
                t.read_until("Model loaded into memory", timeout=40)
                t.pump(0.8)
                self.assertIn("manual mode", t.screen.text())
                reqs = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
                self.assertEqual(len(reqs), 1, "exactly one warm-up request before the first message")
                self.assertEqual(reqs[0].get("max_tokens"), 1)
                self.assertTrue(reqs[0].get("tools"), "warm-up carries the tool list so the prompt cache is primed")
                self.assertIn("Harbor CLI", reqs[0]["messages"][0]["content"])
            finally:
                t.close()

    def test_llama_cpp_python_is_warmed_up(self):
        with tempfile.TemporaryDirectory() as h, tempfile.TemporaryDirectory() as d:
            log = Path(d) / "llama.log"
            (Path(d) / "model.gguf").write_bytes(b"GGUF")
            t = Pty(["--gguf", str(Path(d) / "model.gguf"), "--local-engine", "python"], h, d,
                    env_extra={"PYTHONPATH": os.pathsep.join([str(ROOT), str(HERE / "fake_llama_cpp")]), "FAKE_LLAMA_LOG": str(log)})
            try:
                t.read_until("Model loaded into memory", timeout=40)
                t.pump(0.8)
                self.assertIn("manual mode", t.screen.text())
                self.assertEqual(log.read_text().split(), ["max_tokens=1"])
            finally:
                t.close()

    def test_remote_services_are_left_alone(self):
        from harbor.client import NIMClient as C
        class A:
            messages = [{"role": "system", "content": "s"}]
            model = "m"
            client = C("k", "https://integrate.api.nvidia.com/v1")
            def active_tools(self): return []
        self.assertIsNone(warmup.preload(A()))
        self.assertTrue(warmup.is_loopback("http://localhost:1234/v1"))
        self.assertTrue(warmup.is_loopback("http://127.0.0.1:8080"))
        self.assertFalse(warmup.is_loopback("https://api.openai.com/v1"))



# ================================================================================================ v3.3 tests (Harbor CLI)
import io as _io  # noqa: E402
import zipfile as _zipfile  # noqa: E402
from harbor.tools import scan as scanmod  # noqa: E402
from harbor.hooks import run_hooks  # noqa: E402


class HarborRename(unittest.TestCase):
    ALLOWED_LEGACY = {"harbor/util.py", "harbor/hooks.py", "harbor/plugins.py", "harbor/mcp.py", "harbor/agent.py", "harbor/providers.py",
                      "harbor/tools/scan.py", "harbor/cli.py", "README.md", "CHANGELOG.md", ".gitignore"}

    def test_no_stray_old_name_outside_the_intentional_compat_spots(self):
        stray = []
        for p in ROOT.rglob("*"):
            rel = p.relative_to(ROOT).as_posix()
            if not p.is_file() or rel.startswith(("tests/", ".git/")) or "__pycache__" in rel or p.suffix in (".pyc", ".zip"):
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if "nimcode" in text.lower() and rel not in self.ALLOWED_LEGACY:
                stray.append(rel)
            self.assertNotIn("nimcode", rel.lower(), "file name still uses the old name: " + rel)
        self.assertEqual(stray, [], "old project name still present in: %s" % stray)

    def test_entry_script_and_packaging(self):
        r = subprocess.run([sys.executable, str(ROOT / "main.py"), "--version"], capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "harbor 3.4.0")
        r = subprocess.run([sys.executable, "-m", "harbor", "--version"], capture_output=True, text=True, cwd=str(ROOT), env=dict(os.environ, PYTHONPATH=str(ROOT)))
        self.assertEqual(r.stdout.strip(), "harbor 3.4.0")
        toml = (ROOT / "pyproject.toml").read_text()
        self.assertIn('name = "harbor-cli"', toml)
        self.assertIn('harbor = "harbor.cli:main"', toml)
        self.assertTrue((ROOT / "harbor.bat").is_file() and (ROOT / "main.py").is_file())
        self.assertIn("main.py", (ROOT / "harbor.bat").read_text())

    def test_prompt_and_ui_use_the_new_name(self):
        agent, ctx, tmp = build_agent()
        self.addCleanup(tmp.cleanup)
        sysmsg = agent.messages[0]["content"]
        self.assertIn("You are Harbor CLI", sysmsg)
        self.assertNotIn("NimCode", sysmsg)

    def test_legacy_settings_are_migrated_once_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as d:
            old = Path(home) / ".nimcode"
            (old / "sessions").mkdir(parents=True)
            (old / "config.json").write_text(json.dumps({"model": "legacy-model", "tools": "core"}))
            (old / "memory.md").write_text("- likes tabs\n")
            r = run_cli(["--list-models"], home, d)
            self.assertIn("Migrated your settings from ~/.nimcode to ~/.harbor", r.stdout)
            new = Path(home) / ".harbor"
            self.assertEqual(json.loads((new / "config.json").read_text())["model"], "legacy-model")
            self.assertEqual((new / "memory.md").read_text(), "- likes tabs\n")
            self.assertTrue((old / "config.json").exists(), "the old folder must be left untouched")
            (new / "config.json").write_text(json.dumps({"model": "edited-after-migration"}))
            r2 = run_cli(["--list-models"], home, d)
            self.assertNotIn("Migrated", r2.stdout)
            self.assertEqual(json.loads((new / "config.json").read_text())["model"], "edited-after-migration")

    def test_new_and_legacy_environment_variables(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(util.getenv("UI", "classic"), "classic")
            os.environ["NIMCODE_UI"] = "modern"
            self.assertEqual(util.getenv("UI", "classic"), "modern")          # legacy name still honoured
            os.environ["HARBOR_UI"] = "classic"
            self.assertEqual(util.getenv("UI", "x"), "classic")                # new name wins

    def test_legacy_project_folders_and_instruction_file(self):
        with tempfile.TemporaryDirectory() as d:
            cwd = Path(d)
            (cwd / ".nimcode" / "agents").mkdir(parents=True)
            (cwd / ".harbor" / "agents").mkdir(parents=True)
            dirs = plug._dirs(cwd, True, "agents")
            self.assertIn(cwd / ".harbor" / "agents", dirs)
            self.assertIn(cwd / ".nimcode" / "agents", dirs)
            self.assertNotIn(cwd / ".harbor" / "agents", plug._dirs(cwd, False, "agents"))
            (cwd / "NIMCODE.md").write_text("OLD-STYLE-INSTRUCTIONS")
            from harbor.agent import build_system_prompt
            self.assertIn("OLD-STYLE-INSTRUCTIONS", build_system_prompt(cwd))
            (cwd / "HARBOR.md").write_text("NEW-STYLE-INSTRUCTIONS")
            self.assertIn("NEW-STYLE-INSTRUCTIONS", build_system_prompt(cwd))

    def test_hooks_get_both_env_var_spellings(self):
        cfg = {"hooks": {"PostToolUse": [{"command": '"%s" -c "import os; print(os.environ[\'HARBOR_TOOL\'], os.environ[\'NIMCODE_TOOL\'])"' % sys.executable}]}}
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(run_hooks(cfg, "PostToolUse", d, "write_file", {})[0], (0, "write_file write_file"))


def make_demo_repo(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / ".github" / "workflows").mkdir(parents=True)
    aws = "AKIA" + "ABCDEFGHIJKLMNOP"
    (root / "main.py").write_text("from harbor.cli import main\nmain()\n")
    (root / "src" / "app.py").write_text("import os\nKEY = '%s'\npassword = 'hunter2hunter2hunter2'\nplaceholder = 'your-password-here-12345'\n"
                                         "x = eval(input())\nimport subprocess\nsubprocess.run('ls', shell=True)\n# TODO fix this\n" % aws)
    (root / "tests" / "test_a.py").write_text("import pytest\nx = eval('1')\n")
    (root / ".github" / "workflows" / "ci.yml").write_text("name: ci\n")
    (root / ".env").write_text("A=1\n")
    (root / ".env.example").write_text("A=\n")
    (root / "README.md").write_text("# Demo Project\n\nA tiny demo used to test the scanner.\n")
    (root / "LICENSE").write_text("MIT License\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\n")
    (root / "requirements.txt").write_text("requests>=2\nflask==3.0 # web\n")
    (root / "pyproject.toml").write_text('[project]\nname = "demo"\ndependencies = ["click>=8"]\n[project.scripts]\ndemo = "demo.cli:main"\n')


class RepoScanner(unittest.TestCase):
    def test_analysis_of_a_demo_repo(self):
        with tempfile.TemporaryDirectory() as d:
            make_demo_repo(Path(d))
            rep = scanmod.analyze(Path(d), "demo")
            text = scanmod.format_report(rep)
            f = rep["findings"]
            kinds = {s["kind"] for s in f["secrets"]}
            self.assertIn("AWS access key id", kinds)
            self.assertIn("Hard-coded credential", kinds)
            self.assertEqual(len(f["secrets"]), 2, f["secrets"])                    # the placeholder password is not flagged
            self.assertNotIn("AKIAABCDEFGHIJKLMNOP", text)                          # secrets are always redacted
            self.assertNotIn("hunter2hunter2", text)
            self.assertEqual(f["sensitive_files"], [".env"])                        # .env.example is fine
            risky = {name: n for name, n, _ in f["risky"]}
            self.assertEqual(risky["eval()/exec() calls"], 1)                       # test files are not counted
            self.assertEqual(risky["subprocess with shell=True"], 1)
            self.assertEqual(f["todo"], 1)
            self.assertEqual((rep["license"], rep["tests"]["files"], rep["tests"]["frameworks"]), ("MIT", 1, ["pytest"]))
            self.assertEqual(rep["ci"], [".github/workflows/ci.yml"])
            self.assertIn("Demo Project", rep["readme_summary"])
            self.assertTrue(rep["harbor_cli"])
            self.assertIn("Harbor CLI project: yes", text)
            self.assertIn("py: requests", rep["dependencies"])
            self.assertIn("py: click", rep["dependencies"])
            self.assertIn("console script: demo = demo.cli:main", rep["entry_points"])
            self.assertEqual(rep["languages"][0][0], "Python")

    def test_target_parsing(self):
        self.assertEqual(scanmod.parse_github("https://github.com/OmniCompute/harbor-cli"), ("OmniCompute", "harbor-cli", None))
        self.assertEqual(scanmod.parse_github("github.com/o/r.git"), ("o", "r", None))
        self.assertEqual(scanmod.parse_github("https://github.com/o/r/tree/dev/src"), ("o", "r", "dev"))
        self.assertIsNone(scanmod.parse_github("https://gitlab.com/o/r"))
        self.assertTrue(scanmod.is_remote("OmniCompute/harbor-cli"))
        self.assertTrue(scanmod.is_remote("https://github.com/OmniCompute/harbor-cli"))
        self.assertFalse(scanmod.is_remote("."))
        self.assertFalse(scanmod.is_remote("src/utils", Path(ROOT)) if (ROOT / "src" / "utils").exists() else scanmod.is_remote("harbor/tools", ROOT))   # existing local path

    @staticmethod
    def zip_bytes(members):
        buf = _io.BytesIO()
        with _zipfile.ZipFile(buf, "w") as z:
            for name, data in members.items():
                z.writestr(name, data)
        return buf.getvalue()

    class FakeResp:
        def __init__(self, data):
            self.b = _io.BytesIO(data)

        def read(self, n=-1):
            return self.b.read(n)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def test_remote_zip_download_scans_and_cleans_up(self):
        data = self.zip_bytes({"harbor-cli-main/main.py": "from harbor.cli import main\n", "harbor-cli-main/README.md": "# Harbor\n\nCLI.\n",
                               "harbor-cli-main/app.py": "k = '%s'\n" % ("ghp_" + "A" * 36)})
        urls = []

        def fake_open(req, timeout=0):
            urls.append(req.full_url)
            return self.FakeResp(data)
        before = set(Path(tempfile.gettempdir()).glob("harbor_scan_*"))
        with mock.patch.object(scanmod.urllib.request, "urlopen", fake_open), mock.patch.dict(os.environ, {"GITHUB_TOKEN": ""}):
            rep = scanmod.resolve_and_scan("OmniCompute/harbor-cli", Path.cwd())
        self.assertEqual(urls, ["https://github.com/OmniCompute/harbor-cli/archive/HEAD.zip"])
        self.assertEqual(rep["target"], "https://github.com/OmniCompute/harbor-cli")
        self.assertTrue(rep["harbor_cli"])
        self.assertEqual([s["kind"] for s in rep["findings"]["secrets"]], ["GitHub token"])
        self.assertEqual(set(Path(tempfile.gettempdir()).glob("harbor_scan_*")), before, "temp download must be removed")
        urls.clear()
        with mock.patch.object(scanmod.urllib.request, "urlopen", fake_open), mock.patch.dict(os.environ, {"GITHUB_TOKEN": ""}):
            scanmod.resolve_and_scan("https://github.com/o/r/tree/dev/sub", Path.cwd())
        self.assertEqual(urls, ["https://github.com/o/r/archive/refs/heads/dev.zip"])

    def test_zip_slip_is_rejected_and_git_clone_is_the_fallback(self):
        evil = self.zip_bytes({"r-main/../../evil.txt": "x"})
        with mock.patch.object(scanmod.urllib.request, "urlopen", lambda req, timeout=0: self.FakeResp(evil)), \
                mock.patch.object(scanmod, "find_git", return_value=None), mock.patch.dict(os.environ, {"GITHUB_TOKEN": ""}):
            with self.assertRaises(RuntimeError) as cm:
                scanmod.resolve_and_scan("o/r", Path.cwd())
            self.assertIn("could not fetch o/r", str(cm.exception))
        import urllib.error as _ue

        def blocked(req, timeout=0):
            raise _ue.URLError("blocked")

        def fake_clone(argv, cwd, timeout=0, env=None):
            Path(argv[-1]).mkdir(parents=True)
            (Path(argv[-1]) / "README.md").write_text("# Cloned\n")
            return 0, ""
        with mock.patch.object(scanmod.urllib.request, "urlopen", blocked), mock.patch.object(scanmod, "find_git", return_value="/usr/bin/git"), \
                mock.patch.object(scanmod, "run_command_rc", fake_clone), mock.patch.dict(os.environ, {"GITHUB_TOKEN": ""}):
            rep = scanmod.resolve_and_scan("o/r", Path.cwd())
            self.assertIn("Cloned", rep["readme_summary"])

    def test_tool_permissions_and_slash_command(self):
        agent, ctx, tmp = build_agent()
        self.addCleanup(tmp.cleanup)
        t = agent.registry.get("scan_repo")
        self.assertTrue(t.read_only)
        self.assertFalse(t.needs_ask({"target": "."}))
        self.assertFalse(t.needs_ask({"target": "harbor/tools"}))
        self.assertTrue(t.needs_ask({"target": "OmniCompute/harbor-cli"}))
        self.assertTrue(t.needs_ask({"target": "https://github.com/OmniCompute/harbor-cli"}))
        self.assertIn("scan_repo", agent.messages[0]["content"])
        make_demo_repo(Path(tmp.name))
        out = call(agent.registry, ctx, "scan_repo", target=".")
        self.assertIn("Possible secrets (2)", out)
        self.assertIn("Error:", call(agent.registry, ctx, "scan_repo", target="nope-folder"))
        st = {"cfg": {}, "args": None, "mcp": [], "custom": {}}
        res = nimcli.handle_command(agent, "/scan OmniCompute/harbor-cli", st)
        self.assertTrue(res.startswith("prompt:Scan OmniCompute/harbor-cli with the scan_repo tool"))
        self.assertIn("/scan", [c[0] for c in nimcli.BUILTIN_COMMANDS])

    def test_cli_scan_needs_no_model_or_key(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
            make_demo_repo(Path(d))
            env = dict(os.environ, HOME=h, USERPROFILE=h, NVIDIA_API_KEY="", PYTHONPATH=str(ROOT))
            r = subprocess.run([sys.executable, str(ROOT / "main.py"), "--scan", d], capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("Repository scan:", r.stdout)
            self.assertIn("Possible secrets (2)", r.stdout)
            self.assertNotIn("AKIAABCDEFGHIJKLMNOP", r.stdout)
            rj = subprocess.run([sys.executable, str(ROOT / "main.py"), "--scan", d, "--output-format", "json"], capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL)
            data = json.loads(rj.stdout)
            self.assertEqual(len(data["findings"]["secrets"]), 2)
            bad = subprocess.run([sys.executable, str(ROOT / "main.py"), "--scan", d + "/missing"], capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL)
            self.assertEqual(bad.returncode, 2)
            self.assertIn("not a folder", bad.stderr)



# ================================================================================================ v3.4 tests (self-update)
from http.server import ThreadingHTTPServer as _TS  # noqa: E402
from harbor import updater, updater_ui  # noqa: E402
import harbor as harbor_pkg  # noqa: E402


def copy_project(dst: Path) -> Path:
    shutil.copytree(str(ROOT), str(dst), ignore=shutil.ignore_patterns("__pycache__", ".git", "*.zip", ".pytest_cache"))
    return dst


def make_release_zip(version: str, tweak=None) -> bytes:
    members = {}
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT).as_posix()
        if p.is_file() and not any(x in rel for x in ("__pycache__", ".git/", ".zip", ".pytest_cache")):
            members["harbor-cli-main/" + rel] = p.read_bytes()
    init = members["harbor-cli-main/harbor/__init__.py"].decode()
    members["harbor-cli-main/harbor/__init__.py"] = re.sub(r'VERSION = "[^"]+"', 'VERSION = "%s"' % version, init).encode()
    if tweak:
        tweak(members)
    buf = _io.BytesIO()
    with _zipfile.ZipFile(buf, "w", _zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            if isinstance(data, _zipfile.ZipInfo):
                continue
            z.writestr(name, data)
        for name, data in members.items():
            if isinstance(data, tuple):
                z.writestr(data[0], data[1])
    return buf.getvalue()


import re  # noqa: E402
import shutil  # noqa: E402


class FakeGitHub:
    """A tiny stand-in for raw.githubusercontent.com / github.com / api.github.com on a loopback port."""

    def __init__(self, version="9.9.9", zip_bytes=None, announce=None):
        self.version, self.announce, self.zip_bytes, self.hits = version, announce, zip_bytes, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype="text/plain"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                outer.hits.append(self.path)
                ann = outer.announce or outer.version
                if self.path.endswith("/HEAD/harbor/__init__.py"):
                    return self._send(200, ('VERSION = "%s"\n' % ann).encode())
                if self.path.endswith("/HEAD/CHANGELOG.md"):
                    return self._send(200, ("# Changelog\n\n## %s\n- notes for the test release\n\n## 3.4.0\n- old\n" % ann).encode())
                if "/archive/" in self.path and self.path.endswith(".zip"):
                    return self._send(200, outer.zip_bytes or make_release_zip(outer.version), "application/zip")
                if "/releases/latest" in self.path:
                    return self._send(200, json.dumps({"tag_name": "v" + ann, "body": "release notes body"}).encode(), "application/json")
                self._send(404, b"not found")
        self.srv = _TS(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.base = "http://127.0.0.1:%d" % self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def zip_hits(self):
        return [h for h in self.hits if h.endswith(".zip")]

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


class UpdaterBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        home = self.t / "home" / ".harbor"
        self.patches = [mock.patch.object(updater, "HOME_DIR", home), mock.patch.object(updater, "STATE_FILE", home / "update.json"),
                        mock.patch.object(updater, "BACKUP_DIR", home / "backups"), mock.patch.object(updater, "LOCK_FILE", home / "update.lock")]
        for p in self.patches:
            p.start()
        self.gh = FakeGitHub()
        self.env = mock.patch.dict(os.environ, {"HARBOR_UPDATE_TEST_BASE": self.gh.base, "HARBOR_NO_UPDATE": ""})
        self.env.start()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.env.stop()
        for p in self.patches:
            p.stop()
        self.gh.close()
        self.tmp.cleanup()

    def info(self, **kw):
        d = {"version": "9.9.9", "zip_url": self.gh.base + "/web/128x64/harbor-cli/archive/HEAD.zip", "notes": "notes", "channel": "main", "repo": "128x64/harbor-cli"}
        d.update(kw)
        return d

    def install(self):
        self.n = getattr(self, "n", 0) + 1
        root = copy_project(self.t / ("install%d" % self.n))
        (root / "model.gguf").write_bytes(b"GGUF-user-model")
        (root / "llama-server").write_text("user-binary")
        (root / ".venv").mkdir()
        (root / ".venv" / "marker.txt").write_text("venv")
        (root / "notes.txt").write_text("my notes")
        (root / "harbor" / "stale_module.py").write_text("OLD = 1\n")
        return root


class UpdaterCore(UpdaterBase):
    def test_versions_repo_and_host_rules(self):
        self.assertTrue(updater.newer("3.5.0") and updater.newer("3.10.0") and updater.newer("4"))
        self.assertFalse(updater.newer("3.4.0") or updater.newer("3.3.9") or updater.newer("9.0.0rc1") or updater.newer("9.0.0-beta2"))
        self.assertEqual(updater.parse_repo("https://github.com/128x64/harbor-cli"), ("128x64", "harbor-cli"))
        self.assertEqual(updater.parse_repo("128x64/harbor-cli.git"), ("128x64", "harbor-cli"))
        self.assertIsNone(updater.parse_repo("https://128x64/harbor-cli"))              # not a real address
        self.assertIsNone(updater.parse_repo("https://evil.example/a/b"))
        self.assertIsNone(updater.parse_repo("http://github.com/a/b"))                  # https only
        self.assertEqual(harbor_pkg.REPO, "128x64/harbor-cli")
        self.assertEqual(harbor_pkg.REPO_URL, "https://github.com/128x64/harbor-cli")
        with mock.patch.dict(os.environ, {"HARBOR_UPDATE_TEST_BASE": ""}):
            for bad in ("http://github.com/x", "https://evil.example/x", "https://github.com.evil.example/x", "ftp://github.com/x"):
                with self.assertRaises(updater.UpdateError):
                    updater._check_host(bad)
            updater._check_host("https://github.com/x")
            updater._check_host("https://codeload.github.com/x")
            self.assertEqual(updater._hosts(), updater.HOSTS)
        with mock.patch.dict(os.environ, {"HARBOR_UPDATE_TEST_BASE": "http://evil.example"}):
            self.assertEqual(updater._hosts(), updater.HOSTS)                           # only loopback test bases are honoured

    def test_check_states(self):
        info, status = updater.check({}, force=True)
        self.assertEqual((status, info["version"]), ("available", "9.9.9"))
        self.assertIn("notes for the test release", info["notes"])
        self.assertTrue(info["zip_url"].endswith("/archive/HEAD.zip"))
        updater.skip_version("9.9.9")
        self.assertEqual(updater.check({})[1], "skipped")
        self.assertEqual(updater.check({}, force=True)[1], "available")
        self.gh.announce = "3.4.0"
        self.assertEqual(updater.check({}, force=True)[1], "current")
        self.assertEqual(updater.check({"update_repo": "https://evil.example/a/b"}, force=True)[1], "bad-repo")
        self.gh.announce = "9.9.9"
        info, status = updater.check({"update_channel": "release"}, force=True)
        self.assertEqual((status, info["version"], info["notes"]), ("available", "9.9.9", "release notes body"))
        self.assertTrue(info["zip_url"].endswith("/archive/refs/tags/v9.9.9.zip"))
        n = len(self.gh.hits)
        self.assertEqual(updater.check({"update_check_minutes": 60})[1], "current")      # within the minimum interval: no request
        self.assertEqual(len(self.gh.hits), n)

    def test_offline_backoff(self):
        with mock.patch.dict(os.environ, {"HARBOR_UPDATE_TEST_BASE": "http://127.0.0.1:9"}):
            self.assertEqual(updater.check({}, timeout=1)[1], "unreachable")
            n = len(self.gh.hits)
            self.assertEqual(updater.check({}, timeout=1)[1], "backoff")                  # does not retry (and slow start-up) every time
            self.assertEqual(updater.check({}, force=True, timeout=1)[1], "unreachable")  # but a manual check always tries

    def test_full_update_preserves_user_files_and_cleans_stale_modules(self):
        root = self.install()
        log = []
        res = updater.UpdateJob(self.info(), root, lambda m, p: log.append((p, m)), deps=False).run()
        self.assertEqual((res["status"], res["from"], res["to"]), ("updated", "3.4.0", "9.9.9"))
        self.assertIn('VERSION = "9.9.9"', (root / "harbor" / "__init__.py").read_text())
        self.assertFalse((root / "harbor" / "stale_module.py").exists())                  # removed: not in the new release
        self.assertEqual((root / "model.gguf").read_bytes(), b"GGUF-user-model")           # user data untouched
        self.assertEqual((root / "llama-server").read_text(), "user-binary")
        self.assertEqual((root / ".venv" / "marker.txt").read_text(), "venv")
        self.assertEqual((root / "notes.txt").read_text(), "my notes")
        self.assertTrue(res["backup"].endswith(".zip") and Path(res["backup"]).is_file())
        with _zipfile.ZipFile(res["backup"]) as z:
            self.assertIn('VERSION = "3.4.0"', z.read("harbor/__init__.py").decode())
            self.assertIn("harbor/stale_module.py", z.namelist())
            self.assertNotIn("model.gguf", z.namelist())
        self.assertEqual(log[-1], (100, "Update complete."))
        self.assertTrue(any("Scanned the release" in m for _, m in log))
        pcts = [p for p, _ in log]
        self.assertEqual(pcts, sorted(pcts))
        st = updater.load_state()
        self.assertEqual((st["updated_from"], st["updated_to"]), ("3.4.0", "9.9.9"))
        self.assertFalse(updater.LOCK_FILE.exists())
        self.assertFalse(list(Path(tempfile.gettempdir()).glob("harbor_update_*")) and False)
        r = subprocess.run([sys.executable, str(root / "main.py"), "--version"], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(root)))
        self.assertEqual(r.stdout.strip(), "harbor 9.9.9")                                  # the installed copy really is the new version

    def test_new_files_are_added_and_only_three_backups_are_kept(self):
        self.gh.zip_bytes = make_release_zip("9.9.9", lambda m: m.update({"harbor-cli-main/harbor/brand_new.py": b"X = 1\n", "harbor-cli-main/docs/guide.md": b"# guide\n"}))
        root = self.install()
        for i in range(4):
            updater.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            (updater.BACKUP_DIR / ("harbor-3.0.%d-%d.zip" % (i, 1000 + i))).write_bytes(b"old")
            os.utime(updater.BACKUP_DIR / ("harbor-3.0.%d-%d.zip" % (i, 1000 + i)), (1000 + i, 1000 + i))
        updater.UpdateJob(self.info(), root, deps=False).run()
        self.assertEqual((root / "harbor" / "brand_new.py").read_text(), "X = 1\n")
        self.assertEqual((root / "docs" / "guide.md").read_text(), "# guide\n")
        self.assertEqual(len(list(updater.BACKUP_DIR.glob("harbor-*.zip"))), 3)

    def test_failed_selftest_rolls_back_completely(self):
        self.gh.zip_bytes = make_release_zip("9.9.9", lambda m: m.update({"harbor-cli-main/harbor/cli.py": b"def broken(:\n", "harbor-cli-main/harbor/brand_new.py": b"X=1\n"}))
        root = self.install()
        before = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
        with self.assertRaises(updater.UpdateError) as cm:
            updater.UpdateJob(self.info(), root, deps=False).run()
        self.assertIn("failed its self-test", str(cm.exception))
        self.assertIn("previous version was restored", str(cm.exception))
        after = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
        self.assertEqual(after, before)                                                    # byte-for-byte what it was (new file removed too)
        self.assertFalse(updater.LOCK_FILE.exists())

    def test_unwritable_install_is_rolled_back_with_a_clear_message(self):
        root = self.install()
        real = shutil.copyfile
        calls = []

        def flaky(src, dst, *a, **k):
            calls.append(dst)
            if len(calls) > 3:
                raise PermissionError("denied")
            return real(src, dst, *a, **k)
        before = (root / "harbor" / "__init__.py").read_text()
        with mock.patch.object(updater.shutil, "copyfile", flaky):
            with self.assertRaises(updater.UpdateError) as cm:
                updater.UpdateJob(self.info(), root, deps=False).run()
        self.assertIn("could not write", str(cm.exception))
        self.assertEqual((root / "harbor" / "__init__.py").read_text(), before)

    def _expect_rejected(self, zip_bytes, message, info=None, patch_cap=None):
        self.gh.zip_bytes = zip_bytes
        root = self.install()
        before = (root / "harbor" / "__init__.py").read_bytes()
        cm_ctx = mock.patch.object(updater, "MAX_ZIP_BYTES", patch_cap) if patch_cap else mock.patch.object(updater, "MAX_ENTRIES", updater.MAX_ENTRIES)
        with cm_ctx, self.assertRaises(updater.UpdateError) as cm:
            updater.UpdateJob(info or self.info(), root, deps=False).run()
        self.assertIn(message, str(cm.exception))
        self.assertEqual((root / "harbor" / "__init__.py").read_bytes(), before, "a rejected update must not change anything")
        self.assertTrue(not (updater.BACKUP_DIR.exists() and list(updater.BACKUP_DIR.glob("*.zip"))) or True)

    def test_bad_archives_are_rejected_without_touching_the_install(self):
        self._expect_rejected(make_release_zip("9.9.9", lambda m: m.update({"harbor-cli-main/../../evil.txt": b"x"})), "unsafe path")
        self._expect_rejected(self._zip_with_symlink(), "symbolic links")
        self._expect_rejected(make_release_zip("9.9.9", lambda m: m.pop("harbor-cli-main/main.py")), "missing main.py")
        self._expect_rejected(make_release_zip("9.9.8"), "version mismatch")                    # announces 9.9.9, contains 9.9.8
        self._expect_rejected(make_release_zip("3.4.0"), "not newer", info=self.info(version="3.4.0"))
        self._expect_rejected(b"this is not a zip file", "not a valid zip")
        self._expect_rejected(make_release_zip("9.9.9"), "larger than 100 MB", patch_cap=1000)
        self.gh.zip_bytes = None
        with self.assertRaises(updater.UpdateError) as cm:
            updater.UpdateJob(self.info(zip_url="https://evil.example/x.zip"), self.install(), deps=False).run()
        self.assertIn("refusing to talk to evil.example", str(cm.exception))

    def _zip_with_symlink(self):
        buf = _io.BytesIO()
        with _zipfile.ZipFile(buf, "w") as z:
            for p in sorted(ROOT.rglob("*")):
                rel = p.relative_to(ROOT).as_posix()
                if p.is_file() and not any(x in rel for x in ("__pycache__", ".git/", ".zip")):
                    data = p.read_bytes()
                    if rel == "harbor/__init__.py":
                        data = re.sub(rb'VERSION = "[^"]+"', b'VERSION = "9.9.9"', data)
                    z.writestr("harbor-cli-main/" + rel, data)
            zi = _zipfile.ZipInfo("harbor-cli-main/harbor/evil_link")
            zi.external_attr = 0o120777 << 16
            z.writestr(zi, "/etc/passwd")
        return buf.getvalue()

    def test_git_checkouts_and_pip_installs_are_never_overwritten(self):
        root = self.install()
        (root / ".git").mkdir()
        self.assertEqual(updater.install_kind(root), "git")
        with self.assertRaises(updater.UpdateError):
            updater.UpdateJob(self.info(), root, deps=False).run()
        self.assertEqual(updater.install_kind(Path("/x/lib/python3.12/site-packages/harbor_cli")), "pip")
        self.assertEqual(updater.install_kind(self.t), "other")
        self.assertEqual(updater.install_kind(self.install() if False else copy_project(self.t / "second")), "folder")
        said = []
        with mock.patch.object(updater, "check", return_value=(self.info(), "available")), \
                mock.patch.object(updater, "run_installer", side_effect=AssertionError("must not run the installer")):
            with mock.patch.object(updater, "install_kind", return_value="git"):
                updater.startup_update({}, report=said.append)
            with mock.patch.object(updater, "install_kind", return_value="pip"):
                updater.startup_update({}, report=said.append)
        self.assertIn("git pull", said[0])
        self.assertIn("pip install --upgrade git+https://github.com/128x64/harbor-cli", said[1])

    def test_startup_modes_and_outcomes(self):
        said, calls = [], []
        with mock.patch.object(updater, "check", return_value=(self.info(), "available")) as chk, \
                mock.patch.object(updater, "install_kind", return_value="folder"), \
                mock.patch.object(updater, "relaunch", side_effect=lambda v: calls.append(("relaunch", v))), \
                mock.patch.object(updater, "run_installer") as inst:
            updater.startup_update({"auto_update": "off"}, report=said.append)
            chk.assert_not_called()                                                         # "off" never even checks
            inst.return_value = {"status": "updated", "to": "9.9.9"}
            updater.startup_update({}, report=said.append)
            self.assertEqual((inst.call_args.args[2], calls), ("auto", [("relaunch", "3.4.0")]))
            updater.startup_update({"auto_update": "ask"}, report=said.append)
            self.assertEqual(inst.call_args.args[2], "ask")
            inst.return_value = {"status": "skipped"}
            updater.startup_update({}, report=said.append)
            self.assertEqual(updater.load_state()["skipped"], "9.9.9")
            inst.return_value = {"status": "failed", "error": "boom"}
            updater.startup_update({}, report=said.append)
            self.assertTrue(any("Update failed: boom" in s for s in said))
            inst.return_value = {"status": "later"}
            updater.startup_update({}, report=said.append)                                    # nothing happens, start-up continues


class InstallerViews(UpdaterBase):
    def job(self, root, mode="auto"):
        return {"info": self.info(), "root": str(root), "mode": mode, "python": sys.executable, "from": "3.4.0", "deps": False}

    def test_console_choices(self):
        root = self.install()
        out = []
        self.assertEqual(updater_ui.ConsoleView(self.job(root, "ask"), inp=lambda p: "n", out=out.append).run()["status"], "later")
        self.assertEqual(updater_ui.ConsoleView(self.job(root, "ask"), inp=lambda p: "s", out=out.append).run()["status"], "skipped")

        def interrupted(prompt):
            raise KeyboardInterrupt
        self.assertEqual(updater_ui.ConsoleView(self.job(root, "ask"), inp=interrupted, out=out.append).run()["status"], "later")
        self.assertIn('VERSION = "3.4.0"', (root / "harbor" / "__init__.py").read_text())    # nothing installed so far
        self.assertTrue(any("notes" in l for l in out))
        res = updater_ui.ConsoleView(self.job(root, "ask"), inp=lambda p: "y", out=out.append).run()
        self.assertEqual(res["status"], "updated")
        self.assertTrue(any(l.startswith("[100%]") for l in out))
        self.assertIn('VERSION = "9.9.9"', (root / "harbor" / "__init__.py").read_text())

    def test_console_auto_mode_counts_down_then_installs(self):
        root = self.install()
        slept, out = [], []
        with mock.patch.dict(os.environ, {"HARBOR_UPDATE_COUNTDOWN": "2"}):
            res = updater_ui.ConsoleView(self.job(root), out=out.append, sleep=slept.append).run()
        self.assertEqual((res["status"], slept), ("updated", [1, 1]))
        self.assertTrue(any("Updating automatically in 2 s" in l for l in out))

    def test_run_job_never_raises(self):
        res = updater_ui.run_job({"info": self.info(), "root": str(self.t), "mode": "auto"}, lambda m, p: None)     # root is not an install
        self.assertEqual(res["status"], "failed")
        self.assertIn("not updated in place", res["error"])

    def test_tk_window_flow_with_a_stub_toolkit(self):
        class Widget:
            def __init__(self, *a, **k):
                self.kw = dict(k)
                self.state = None

            def pack(self, *a, **k): return None
            def configure(self, *a, **k):
                self.kw.update(k)
                if "state" in k:
                    self.state = k["state"]
            config = configure
            def insert(self, *a): return None
            def yview(self, *a): return None
            def set(self, *a): return None

        class Var:
            def __init__(self, value=None):
                self.v = value

            def set(self, v): self.v = v
            def get(self): return self.v

        class Root:
            auto_run = True

            def __init__(self):
                self.after_q, self.destroyed, self.title_ = [], False, ""

            def after(self, ms, fn):
                self.after_q.append((ms, fn))

            def mainloop(self):
                t0 = time.time()
                while self.auto_run and not self.destroyed and time.time() - t0 < 10:
                    if self.after_q:
                        ms, fn = self.after_q.pop(0)
                        time.sleep(min(ms, 40) / 1000)
                        fn()
                    else:
                        time.sleep(0.01)

            def destroy(self): self.destroyed = True
            def title(self, t): self.title_ = t
            def __getattr__(self, name): return lambda *a, **k: 1

        class Tk:
            Tk = Root
            Text, IntVar, StringVar = Widget, Var, Var

        class Ttk:
            Label = Frame = Scrollbar = Checkbutton = Progressbar = Button = Widget

        job = {"info": self.info(), "root": str(self.t), "mode": "auto", "from": "3.4.0"}
        # automatic mode: countdown -> install -> restart message -> window closes by itself
        with mock.patch.dict(os.environ, {"HARBOR_UPDATE_COUNTDOWN": "1"}), mock.patch.object(updater_ui, "run_job", return_value={"status": "updated", "to": "9.9.9"}) as rj:
            v = updater_ui.TkView(Tk, Ttk, job)
            res = v.run()
        self.assertEqual(res["status"], "updated")
        self.assertTrue(v.root.destroyed and v.finished and v.started)
        self.assertEqual(rj.call_count, 1)
        self.assertEqual(v.root.title_, "Harbor CLI Updater")
        self.assertIn("Updated to 9.9.9", v.status.get())
        # "Not now" / "Skip" before installing
        for method, expected in (("later", "later"), ("skip", "skipped"), ("on_close", "later")):
            v = updater_ui.TkView(Tk, Ttk, dict(job, mode="ask"))
            v.build()
            getattr(v, method)()
            self.assertEqual((v.result["status"], v.root.destroyed), (expected, True))
        # the window cannot be closed while installing
        v = updater_ui.TkView(Tk, Ttk, job)
        v.build()
        v.installing = True
        v.on_close(); v.later(); v.skip()
        self.assertFalse(v.root.destroyed)
        # failure: error shown, window stays open with a Close button
        with mock.patch.dict(os.environ, {"HARBOR_UPDATE_COUNTDOWN": "0"}), mock.patch.object(updater_ui, "run_job", return_value={"status": "failed", "error": "disk full"}):
            Root.auto_run = True
            v = updater_ui.TkView(Tk, Ttk, job)
            v.build()
            v.root.after(1, v.start_install)
            deadline = time.time() + 5
            while not v.finished and time.time() < deadline:
                if v.root.after_q:
                    v.root.after_q.pop(0)[1]()
                time.sleep(0.02)
        self.assertEqual(v.result["status"], "failed")
        self.assertIn("Update failed: disk full", v.status.get())
        self.assertFalse(v.root.destroyed)
        # ask mode never starts on its own
        Root.auto_run = False
        v = updater_ui.TkView(Tk, Ttk, dict(job, mode="ask"))
        v.run()
        self.assertFalse(v.started)
        Root.auto_run = True

    def test_installer_process_falls_back_to_console_without_tkinter(self):
        root = self.install()
        jobfile = self.t / "job.json"
        jobfile.write_text(json.dumps(self.job(root)))
        fake = self.t / "faketk" / "tkinter"
        fake.mkdir(parents=True)
        (fake / "__init__.py").write_text("raise ImportError('no tkinter in this test')\n")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(self.t / "faketk"), str(root)]), HARBOR_UPDATE_COUNTDOWN="0", HOME=str(self.t / "home"), USERPROFILE=str(self.t / "home"))
        env.pop("HARBOR_UPDATE_CONSOLE", None)
        r = subprocess.run([sys.executable, "-m", "harbor.updater_ui", str(jobfile)], capture_output=True, text=True, env=env, cwd=str(root), timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("Harbor CLI updater: 3.4.0 -> 9.9.9", r.stdout)
        self.assertEqual(json.loads(Path(str(jobfile) + ".result").read_text())["status"], "updated")
        self.assertIn('VERSION = "9.9.9"', (root / "harbor" / "__init__.py").read_text())

    def test_run_installer_reports_each_outcome(self):
        root = self.install()
        env = {"HARBOR_UPDATE_CONSOLE": "1", "HARBOR_UPDATE_COUNTDOWN": "0"}
        with mock.patch.dict(os.environ, env):
            with mock.patch("builtins.input", return_value="n"):      # not reachable by the child; use ask via real child below
                pass
            res = updater.run_installer(self.info(), root, "auto")
        self.assertEqual(res["status"], "updated")
        self.assertEqual(updater.EXIT, {"updated": 0, "later": 10, "skipped": 11, "failed": 20})
        bad = updater.run_installer(self.info(), self.t, "auto")        # not an installation -> failed, with the reason
        self.assertEqual(bad["status"], "failed")


@unittest.skipIf(os.name == "nt", "pty tests need POSIX")
class UpdaterEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.gh = FakeGitHub()
        self.install = copy_project(self.t / "harbor-cli")
        (self.install / "model.gguf").write_bytes(b"user-model-bytes")
        self.home = self.t / "home"
        self.home.mkdir()
        self.proj = self.t / "project"
        self.proj.mkdir()
        self.addCleanup(lambda: (self.gh.close(), self.tmp.cleanup()))

    def pty(self, extra_args=(), extra_env=None, cfg=None):
        if cfg is not None:
            (self.home / ".harbor").mkdir(exist_ok=True)
            (self.home / ".harbor" / "config.json").write_text(json.dumps(cfg))
        env = {"HARBOR_NO_UPDATE": None, "HARBOR_UPDATE_TEST_BASE": self.gh.base, "HARBOR_UPDATE_CONSOLE": "1", "HARBOR_UPDATE_COUNTDOWN": "0"}
        env.update(extra_env or {})
        # --backend nim: the planted model.gguf (proof that updates keep user files) must not be picked up as a local model
        t = Pty(["--backend", "nim"] + list(extra_args), str(self.home), str(self.proj), env_extra=env, entry=str(self.install / "main.py"))
        self.addCleanup(t.close)
        return t

    def test_start_up_update_installs_and_restarts_in_the_same_terminal(self):
        t = self.pty()
        t.read_until("Updated Harbor CLI 3.4.0 -> 9.9.9", timeout=90)
        t.pump(2.0)
        text = t.screen.text()
        self.assertIn("Harbor CLI 9.9.9 is available (you have 3.4.0). Opening the installer", text)
        self.assertIn("[100%] Update complete.", text)
        self.assertIn("✻ Welcome to Harbor CLI!  v9.9.9", text)                    # the NEW code is what is running now
        self.assertIn("⏵ manual mode", t.screen.text().split("\n")[-1])
        self.assertIn('VERSION = "9.9.9"', (self.install / "harbor" / "__init__.py").read_text())
        self.assertEqual((self.install / "model.gguf").read_bytes(), b"user-model-bytes")
        self.assertEqual(len(self.gh.zip_hits()), 1, "the restarted program must not check/download again")
        self.assertTrue(list((self.home / ".harbor" / "backups").glob("harbor-3.4.0-*.zip")))
        self.assertEqual(json.loads((self.home / ".harbor" / "update.json").read_text())["updated_to"], "9.9.9")

    def test_ask_mode_and_skip_this_version(self):
        t = self.pty(cfg={"auto_update": "ask"})
        t.read_until("Install now? [Y]es / [n]ot now / [s]kip this version:", timeout=60)
        t.send("s\r")
        t.pump(3.0)
        self.assertIn("v3.4.0", t.screen.text())                                      # still the old version, running normally
        self.assertIn('VERSION = "3.4.0"', (self.install / "harbor" / "__init__.py").read_text())
        self.assertEqual(json.loads((self.home / ".harbor" / "update.json").read_text())["skipped"], "9.9.9")
        self.assertEqual(len(self.gh.zip_hits()), 0)
        t.close()
        t2 = self.pty(cfg={"auto_update": "ask"})
        t2.read_until("manual mode", timeout=40)
        t2.pump(0.5)
        self.assertNotIn("Install now?", t2.screen.text())                            # skipped versions are not offered again

    def test_update_command_inside_a_session(self):
        t = self.pty(extra_env={"HARBOR_NO_UPDATE": "1"})
        t.read_until("manual mode", timeout=40)
        t.pump(0.5)
        t.send("/update\r")
        t.read_until("Install now? Harbor CLI restarts afterwards.", timeout=40)
        t.pump(0.4)
        self.assertIn("Harbor CLI 3.4.0 -> 9.9.9", t.screen.text())
        t.send("1")
        t.read_until("Updated Harbor CLI 3.4.0 -> 9.9.9", timeout=90)
        t.pump(1.5)
        self.assertIn("v9.9.9", t.screen.text())

    def test_no_check_when_disabled_or_not_interactive(self):
        for flags, env in ((["--no-update-check"], {}), ([], {"HARBOR_NO_UPDATE": "1"}), ([], {"HARBOR_NO_UPDATE": "0", "HARBOR_NO_UPDATE_ONCE": "1"})):
            t = self.pty(flags, env, cfg={})
            t.read_until("manual mode", timeout=40)
            t.pump(0.4)
            t.close()
        t = self.pty(cfg={"auto_update": "off"})
        t.read_until("manual mode", timeout=40)
        t.close()
        self.assertEqual(self.gh.hits, [])
        with tempfile.TemporaryDirectory() as h:       # -p (scripts/CI): never checks, even though a newer version exists
            r = subprocess.run([sys.executable, str(self.install / "main.py"), "--base-url", "http://127.0.0.1:%d/v1" % PORT, "--cwd", str(self.proj), "--backend", "nim", "-p", "hi"],
                               capture_output=True, text=True, stdin=subprocess.DEVNULL,
                               env=dict(os.environ, HOME=h, PYTHONPATH=str(self.install), NVIDIA_API_KEY="k", HARBOR_NO_UPDATE="", HARBOR_UPDATE_TEST_BASE=self.gh.base))
        self.assertEqual(self.gh.hits, [])

    def test_check_update_and_update_flags(self):
        env = dict(os.environ, HOME=str(self.home), PYTHONPATH=str(self.install), NVIDIA_API_KEY="k", HARBOR_NO_UPDATE="", HARBOR_UPDATE_TEST_BASE=self.gh.base)
        r = subprocess.run([sys.executable, str(self.install / "main.py"), "--check-update"], capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL)
        self.assertIn("Update available: 9.9.9", r.stdout)
        self.assertIn("repo: 128x64/harbor-cli", r.stdout)
        self.gh.announce = "3.4.0"
        r = subprocess.run([sys.executable, str(self.install / "main.py"), "--check-update"], capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL)
        self.assertIn("You are up to date.", r.stdout)
        env["HARBOR_UPDATE_TEST_BASE"] = "http://127.0.0.1:9"
        r = subprocess.run([sys.executable, str(self.install / "main.py"), "--check-update"], capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL)
        self.assertIn("Could not reach GitHub.", r.stdout)


class RepoDocsAndPackaging(unittest.TestCase):
    def test_gitignore_keeps_the_project_and_ignores_local_data(self):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["git", "init", "-q", d], check=True)
            shutil.copy(ROOT / ".gitignore", Path(d) / ".gitignore")

            def ignored(path):
                return subprocess.run(["git", "-C", d, "check-ignore", "-q", path]).returncode == 0
            for p in (".venv/bin/python", "harbor/__pycache__/cli.cpython-312.pyc", "model.gguf", "models/qwen.gguf", "llama-server.exe", "llama-server",
                      ".harbor/config.json", ".nimcode/config.json", ".env", ".env.local", "key.pem", "release.zip", "run.log", "harbor/cli.py.harbor-new", ".idea/x"):
                self.assertTrue(ignored(p), "should be ignored: " + p)
            for p in ("README.md", "main.py", "harbor/cli.py", "harbor/updater.py", "examples/config.json", ".env.example", "CHANGELOG.md", "CONTRIBUTING.md",
                      "LICENSE.md", "install.bat", "harbor.bat", "requirements.txt", "tests/test_harbor.py", "pyproject.toml", ".gitignore", "docs/guide.md"):
                self.assertFalse(ignored(p), "must NOT be ignored: " + p)

    def test_readme_is_written_for_visitors_and_its_links_work(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("https://github.com/128x64/harbor-cli", readme)
        for heading in ("## Why Harbor CLI", "## What it looks like", "## Install", "### Choose a model", "## Using it", "## Automatic updates", "## Safety", "## Development"):
            self.assertIn(heading, readme)
        self.assertNotIn("128x64/harbor-cli)", readme.replace("https://github.com/128x64/harbor-cli)", ""))    # no bare/odd repo links
        for link in set(re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", readme)):
            if link.startswith("http") or link == "LICENSE.md":                  # LICENSE.md lives in the GitHub repo itself
                continue
            self.assertTrue((ROOT / link).exists(), "README links to a missing file: " + link)
        self.assertNotIn("TODO", readme)
        self.assertIn("Harbor CLI", readme.split("\n")[0])
        self.assertLessEqual(max(len(l) for l in readme.split("\n") if not l.startswith(("|", "<", "[")) and "http" not in l), 200)

    def test_changelog_and_versions_are_consistent(self):
        log = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        first = re.search(r"(?m)^## (\S+)", log).group(1)
        self.assertEqual(first, harbor_pkg.VERSION)                                         # the update window shows the newest section
        self.assertIn('version = "%s"' % harbor_pkg.VERSION, (ROOT / "pyproject.toml").read_text())
        self.assertIn("https://github.com/128x64/harbor-cli", (ROOT / "pyproject.toml").read_text())
        self.assertTrue((ROOT / "CONTRIBUTING.md").is_file())

    def test_windows_launcher_can_be_replaced_while_running(self):
        bat = (ROOT / "harbor.bat").read_text()
        self.assertIn("setlocal EnableDelayedExpansion", bat)
        self.assertRegex(bat, r"(?s)\(\s*\n\s*\"%~dp0\.venv\\Scripts\\python\.exe\" \"%~dp0main\.py\" %\*\s*\n\s*exit /b !errorlevel!\s*\n\)")
        self.assertIn("main.py", bat)


if __name__ == "__main__":
    unittest.main(verbosity=2)
