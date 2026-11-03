"""Claude Code-style terminal UI: welcome box, streamed markdown, tool call rendering with diffs, spinner with
elapsed time, permission dialogs, and a boxed input editor with slash-command suggestions.

Pure standard library (pygments is used for code highlighting when installed)."""
from __future__ import annotations

import difflib
import os
import random
import re
import shutil
import sys
import threading
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path

from . import util
from .term import EscWatcher, KeyReader, RawMode, flush_input
from .util import IS_WIN

ASCII = os.environ.get("NIMCODE_ASCII") == "1"
if ASCII:
    G = dict(dot="*", elbow="L", spin=["-", "\\", "|", "/"], tl="+", tr="+", bl="+", br="+", h="-", v="|", arrow=">",
             open="[ ]", done="[x]", prog="[~]", bullet="-", star="*", up="^", down="v", plan="||", accept=">>")
else:
    G = dict(dot="●", elbow="⎿", spin=["·", "✢", "✳", "✶", "✻", "✽"], tl="╭", tr="╮", bl="╰", br="╯", h="─", v="│",
             arrow="❯", open="☐", done="☒", prog="☐", bullet="•", star="✻", up="↑", down="↓", plan="⏸", accept="⏵⏵")

VERBS = ["Thinking", "Pondering", "Cogitating", "Computing", "Working", "Analyzing", "Ruminating", "Reasoning", "Brewing", "Synthesizing", "Noodling", "Churning", "Percolating", "Mulling", "Cooking", 
         "Meditating", "Contemplating", "Deliberating", "Reflecting", "Speculating", "Brainstorming", "Processing", "Calculating", "Evaluating", "Examining", "Scrutinizing", "Dissecting", "Investigating", 
         "Exploring", "Probing", "Sifting", "Weighing", "Assessing", "Reviewing", "Studying", "Sleuthing", "Digesting", "Filtering", "Sorting", "Structuring", "Mapping", "Formulating", "Drafting", 
         "Deconstructing", "Assembling", "Compiling", "Aggregating", "Condensing", "Refining", "Polishing", "Tweaking", "Modifying", "Adapting", "Simulating", "Modeling", "Testing", "Verifying", 
         "Validating", "Checking", "Auditing", "Tracking", "Monitoring", "Scanning", "Parsing", "Indexing", "Cataloging", "Decoding", "Decrypting", "Translating", "Interpreting", "Clarifying", 
         "Illuminating", "Unraveling", "Untangling", "Solving", "Resolving", "Figuring", "Determining", "Discovering", "Uncovering", "Detecting", "Diagnosing", "Troubleshooting", "Debugging", 
         "Fixing", "Patching", "Upgrading", "Optimizing", "Maximizing", "Enhancing", "Boosting", "Accelerating", "Streamlining", "Generating", "Creating", "Crafting", "Designing", "Architecting", 
         "Plotting", "Scheming", "Marinating", "Gesticulating", ]

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def sgr(text, code):
    return "\033[%sm%s\033[0m" % (code, text) if util.USE_COLOR else text


def accent(t): return sgr(t, "38;5;173")
def gray(t): return sgr(t, "38;5;245")
def teal(t): return sgr(t, "38;5;73")
def strip_ansi(s): return ANSI_RE.sub("", s)


def cell_len(s: str) -> int:
    n = 0
    for ch in strip_ansi(s):
        if unicodedata.combining(ch):
            continue
        n += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return n


def clip(s: str, width: int) -> str:
    """Truncate an ANSI-colored string to `width` cells (adds an ellipsis)."""
    if cell_len(s) <= width:
        return s
    out, n, i = [], 0, 0
    while i < len(s):
        m = ANSI_RE.match(s, i)
        if m:
            out.append(m.group(0))
            i = m.end()
            continue
        w = 2 if unicodedata.east_asian_width(s[i]) in ("W", "F") else 1
        if n + w > width - 1:
            break
        out.append(s[i])
        n += w
        i += 1
    return "".join(out) + "…" + ("\033[0m" if util.USE_COLOR else "")


def pad(s: str, width: int) -> str:
    return s + " " * max(0, width - cell_len(s))


def term_width() -> int:
    return max(40, shutil.get_terminal_size((100, 30)).columns - 1)


def box(lines, width, color=accent):
    inner = width - 4
    out = [color(G["tl"] + G["h"] * (width - 2) + G["tr"])]
    for ln in lines:
        out.append(color(G["v"]) + " " + pad(clip(ln, inner), inner) + " " + color(G["v"]))
    out.append(color(G["bl"] + G["h"] * (width - 2) + G["br"]))
    return out


# ---------------------------------------------------------------------------- markdown (streaming, line based)
_lexers: dict = {}


def highlight_code(line: str, lang: str) -> str:
    if not util.USE_COLOR:
        return line
    try:
        from pygments import highlight
        from pygments.formatters import Terminal256Formatter
        from pygments.lexers import get_lexer_by_name, TextLexer
        from pygments.util import ClassNotFound
        if lang not in _lexers:
            try:
                _lexers[lang] = get_lexer_by_name(lang or "text", stripnl=False)
            except ClassNotFound:
                _lexers[lang] = TextLexer()
        return highlight(line + "\n", _lexers[lang], Terminal256Formatter(style="monokai")).rstrip("\n")
    except Exception:
        return sgr(line, "38;5;151")


def inline_md(s: str) -> str:
    out = []
    for p in re.split(r"(`[^`\n]+`)", s):
        if len(p) > 2 and p.startswith("`") and p.endswith("`"):
            out.append(sgr(p[1:-1], "38;5;180"))
            continue
        p = re.sub(r"\*\*(.+?)\*\*", lambda m: sgr(m.group(1), "1"), p)
        p = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])", lambda m: sgr(m.group(1), "3"), p)
        p = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", lambda m: m.group(1) + gray(" (" + m.group(2) + ")"), p)
        out.append(p)
    return "".join(out)


class MDStream:
    """Feed text chunks; get fully rendered lines (None = skip) as complete lines arrive."""

    def __init__(self):
        self.buf = ""
        self.in_code = False
        self.lang = ""

    def feed(self, chunk: str):
        self.buf += chunk
        out = []
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            out.append(self.render(line))
        return out

    def flush(self):
        if self.buf:
            line, self.buf = self.buf, ""
            return [self.render(line)]
        return []

    def render(self, line: str):
        line = line.rstrip("\r")
        s = line.strip()
        if s.startswith("```"):
            self.in_code = not self.in_code
            self.lang = s[3:].strip().split()[0] if self.in_code and s[3:].strip() else ""
            return None
        if self.in_code:
            return "  " + highlight_code(line, self.lang)
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            txt = inline_md(m.group(2))
            return sgr(txt, "1;4") if len(m.group(1)) == 1 else sgr(txt, "1")
        if re.match(r"^\s*([-*_])(\s*\1){2,}\s*$", line):
            return gray(G["h"] * 40)
        m = re.match(r"^(\s*)[-*+]\s+(.*)$", line)
        if m:
            return m.group(1) + G["bullet"] + " " + inline_md(m.group(2))
        if s.startswith(">"):
            return gray("▎ ") + sgr(inline_md(s.lstrip("> ")), "3")
        return inline_md(line)


# ---------------------------------------------------------------------------- tool display helpers
DISPLAY = {"read_file": "Read", "write_file": "Write", "edit_file": "Update", "multi_edit": "Update",
           "list_dir": "List", "glob": "Search", "grep": "Search", "web_fetch": "Fetch", "web_search": "Web Search",
           "todo_write": "Update Todos", "task": "Task", "git": "Git", "python_exec": "Python", "sqlite_query": "SQLite",
           "analyze_image": "Vision", "http_request": "HTTP", "ask_user": "Ask", "start_process": "Start Process",
           "process_output": "Process Output", "stop_process": "Stop Process", "finish": "Finish", "delegate": "Delegate",
           "browser_open": "Browser", "browser_text": "Browser Text", "browser_click": "Browser Click", "browser_fill": "Browser Fill"}


def display_name(name: str) -> str:
    if name == "shell":
        from .tools.shell import shell_name
        return {"powershell": "PowerShell", "cmd": "Cmd"}.get(shell_name(), "Bash")
    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        return "%s - %s (MCP)" % (parts[1], parts[2] if len(parts) > 2 else "")
    return DISPLAY.get(name, name.replace("_", " ").title())


def make_diff(old, new, context: int = 3):
    def _lines(t):
        if not t:
            return []
        return (t[:-1] if t.endswith("\n") else t).split("\n")
    a, b = _lines(old), _lines(new)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    rows, adds, dels = [], 0, 0
    for gi, group in enumerate(sm.get_grouped_opcodes(context)):
        if gi:
            rows.append(("sep", 0, ""))
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                rows += [("ctx", j1 + k + 1, a[i1 + k]) for k in range(i2 - i1)]
            else:
                rows += [("del", i1 + k + 1, a[i1 + k]) for k in range(i2 - i1)]
                rows += [("add", j1 + k + 1, b[j1 + k]) for k in range(j2 - j1)]
                dels += i2 - i1
                adds += j2 - j1
    return rows, adds, dels


def render_diff(rows, width, limit=40):
    nw = max([len(str(r[1])) for r in rows] + [1])
    out = []
    for kind, n, text in rows[:limit]:
        text = text.replace("\t", "    ")
        if kind == "sep":
            out.append(gray("  " + "." * 3))
        elif kind == "ctx":
            out.append(gray(clip("  %*d    %s" % (nw, n, text), width)))
        elif kind == "del":
            out.append(sgr(clip("  %*d  - %s" % (nw, n, text), width).ljust(0), "48;5;52;38;5;217"))
        else:
            out.append(sgr(clip("  %*d  + %s" % (nw, n, text), width), "48;5;22;38;5;157"))
    if len(rows) > limit:
        out.append(gray("  ... +%d more lines" % (len(rows) - limit)))
    return out


def plural(n, word):
    return "%d %s%s" % (n, word, "" if n == 1 else "s")


# ---------------------------------------------------------------------------- the UI
class UI:
    def __init__(self):
        self.lock = threading.RLock()
        self.dialog_lock = threading.RLock()
        self._tl = threading.local()
        self.interactive = sys.stdin.isatty() and sys.stdout.isatty()
        self.cwd = Path.cwd()
        self.show_reasoning = False
        self.md = None
        self._first_line = True
        self._last_blank = True
        # spinner
        self._spin_on = False
        self._spin_paused = False
        self._spin_vis = False
        self._spin_label = "Thinking"
        self._spin_t0 = 0.0
        self._spin_tokens = 0
        self._spin_thread = None
        self._spin_i = 0
        self._watch = EscWatcher()
        self._watch_on = False
        self._reason_buf = ""

    @property
    def sub(self) -> bool:
        return getattr(self._tl, "sub", False)

    @sub.setter
    def sub(self, v: bool):
        self._tl.sub = v

    # ------------------------------------------------------------------ low-level output
    def _emit(self, text: str):
        with self.lock:
            if self._spin_vis:
                sys.stdout.write("\r\033[K")
                self._spin_vis = False
            sys.stdout.write(text)
            sys.stdout.flush()
            if text:
                self._last_blank = text.endswith("\n\n") or text.strip() == ""
            if self._spin_on and not self._spin_paused:
                self._draw_spinner()

    def line(self, text: str = ""):
        self._emit(text + "\n")

    def blank(self):
        if not self._last_blank:
            self._emit("\n")

    def rel(self, p) -> str:
        try:
            return str(Path(str(p)).expanduser().resolve().relative_to(self.cwd))
        except Exception:
            return str(p)

    # ------------------------------------------------------------------ spinner
    def _draw_spinner(self):
        if not self.interactive:
            return
        el = int(time.time() - self._spin_t0)
        glyph = G["spin"][self._spin_i % len(G["spin"])]
        info = "%ds" % el
        if self._spin_tokens:
            info += " · %s %s tokens" % (G["down"], ("%.1fk" % (self._spin_tokens / 1000)) if self._spin_tokens >= 1000 else self._spin_tokens)
        txt = "%s %s… (%s · esc to interrupt)" % (glyph, self._spin_label, info)
        sys.stdout.write("\r\033[K" + accent(clip(txt, term_width())))
        sys.stdout.flush()
        self._spin_vis = True

    def _spin_loop(self):
        while self._spin_on:
            time.sleep(0.12)
            with self.lock:
                if self._spin_on and not self._spin_paused:
                    self._spin_i += 1
                    self._draw_spinner()

    def spinner_start(self, label=None):
        if not self.interactive or self.sub or self._spin_on:
            return
        self._spin_label = label or random.choice(VERBS)
        self._spin_t0, self._spin_tokens, self._spin_paused = time.time(), 0, False
        self._spin_on = True
        with self.lock:
            self._draw_spinner()
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

    def spinner_stop(self):
        with self.lock:
            self._spin_on = False
            if self._spin_vis:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
                self._spin_vis = False
        if self._spin_thread:
            self._spin_thread.join(0.5)
            self._spin_thread = None

    def add_tokens(self, n_chars: int):
        self._spin_tokens += max(1, n_chars // 4)

    @contextmanager
    def paused(self):
        """Hide spinner + stop Esc watcher while we need the keyboard (dialogs, ask_user)."""
        with self.lock:
            self._spin_paused = True
            if self._spin_vis:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
                self._spin_vis = False
        was = self._watch_on
        if was:
            self._watch.stop()
        try:
            yield
        finally:
            if was:
                self._watch.start()
            with self.lock:
                self._spin_paused = False

    @contextmanager
    def working(self, label=None):
        self.spinner_start(label)
        if self.interactive and not self.sub:
            self._watch.start()
            self._watch_on = True
        try:
            yield
        finally:
            if self._watch_on:
                self._watch.stop()
                self._watch_on = False
            self.spinner_stop()

    # ------------------------------------------------------------------ session chrome
    def welcome(self, version, model, cwd, notes=()):
        w = min(term_width(), 76)
        lines = [accent(G["star"]) + " " + sgr("Welcome to NimCode!", "1") + gray("  v" + version), "",
                 "  " + gray("/help for help, /status for your current setup"), "",
                 "  " + gray("cwd: ") + str(cwd), "  " + gray("model: ") + model]
        for n in notes:
            lines.append("  " + gray(n))
        for ln in box(lines, w):
            self.line(ln)
        self.line()
        self.line(gray(" Tips for getting started:"))
        for i, t in enumerate(["Ask questions, edit files, or run commands.", "Be specific for the best results.",
                               "Create NIMCODE.md files to customize your interaction.",
                               "Shift+Tab cycles modes; Esc interrupts; /auto <goal> works autonomously; /help lists commands."], 1):
            self.line(gray(" %d. %s" % (i, t)))
        self.line()

    def info(self, text): self.line(gray(text))
    def warn(self, text): self.line(sgr(text, "33"))
    def error(self, text): self.line(sgr(text, "31"))

    def user_echo(self, text: str):
        lines = text.split("\n")
        for i, ln in enumerate(lines):
            self.line(sgr(("> " if i == 0 else "  ") + ln, "48;5;236;1") if util.USE_COLOR else ("> " if i == 0 else "  ") + ln)
        self._last_blank = False

    # ------------------------------------------------------------------ assistant text
    def assistant_text(self, chunk: str):
        if self.sub:
            return
        self.add_tokens(len(chunk))
        if self.md is None:
            self.md, self._first_line = MDStream(), True
        for ln in self.md.feed(chunk):
            self._assistant_line(ln)

    def _assistant_line(self, ln):
        if ln is None:
            return
        if self._first_line:
            if not strip_ansi(ln).strip():
                return
            self.blank()
            self._emit(G["dot"] + " " + ln + "\n")
            self._first_line = False
        else:
            self._emit("  " + ln + "\n")

    def assistant_end(self):
        if self.md is not None:
            for ln in self.md.flush():
                self._assistant_line(ln)
            self.md = None
        if self._reason_buf:
            self._emit(gray("  " + self._reason_buf) + "\n")
            self._reason_buf = ""

    def reasoning(self, chunk: str):
        if self.sub:
            return
        self.add_tokens(len(chunk))
        if not self.show_reasoning:
            return
        self._reason_buf += chunk
        while "\n" in self._reason_buf:
            ln, self._reason_buf = self._reason_buf.split("\n", 1)
            self._emit(sgr("  " + ln, "2;3") + "\n")

    # ------------------------------------------------------------------ tool calls
    def tool_header(self, name: str, summary: str, label=None):
        self.blank()
        label = label or display_name(name)
        if self.sub:
            self._emit(gray("    | %s(%s)" % (label, summary[:80])) + "\n")
            return
        self._emit(G["dot"] + " " + sgr(label, "1") + (gray("(") + summary + gray(")") if summary else "") + "\n")

    def tool_result(self, lines, error=False):
        if self.sub:
            return
        for i, ln in enumerate(lines):
            ln = clip(ln, term_width() - 6)
            if error:
                ln = sgr(strip_ansi(ln), "31")
            self._emit(("  " + G["elbow"] + "  " if i == 0 else "     ") + ln + "\n")

    def show_diff(self, rows, width=None):
        for ln in render_diff(rows, width or term_width() - 6):
            self._emit("     " + ln + "\n")

    def result_lines(self, name: str, args: dict, result: str, todos=None):
        """Return (lines, is_error) summarizing a tool result Claude-Code style."""
        r = result.rstrip()
        err = r.startswith(("Error", "The user denied", "Blocked by"))
        lines = r.splitlines()
        if err:
            return lines[:6] + ([gray("… +%d lines" % (len(lines) - 6))] if len(lines) > 6 else []), True
        if name == "read_file":
            n = len([l for l in lines if re.match(r"^\s*\d+\t", l)]) or len(lines)
            return ["Read " + plural(n, "line")], False
        if name == "list_dir":
            return ["Listed " + plural(len(lines), "entry").replace("entrys", "entries")], False
        if name == "glob":
            return ["No files found" if r.startswith("No files") else "Found " + plural(len(lines), "file")], False
        if name == "grep":
            return ["No matches found" if r.startswith("No matches") else "Found " + plural(len(lines), "match").replace("matchs", "matches")], False
        if name == "web_search":
            return ["Did 1 search" + (" · %s" % plural(len(re.findall(r"^\d+\. ", r, re.M)), "result"))], False
        if name == "web_fetch":
            return ["Fetched %s characters" % format(len(r), ","), gray(lines[0][:100]) if lines else ""], False
        if name == "todo_write" and todos is not None:
            out = []
            for t in todos:
                if t["status"] == "completed":
                    out.append(gray(G["done"] + " " + sgr(t["content"], "9")))
                elif t["status"] == "in_progress":
                    out.append(accent(G["prog"] + " " + sgr(t["content"], "1")))
                else:
                    out.append(G["open"] + " " + t["content"])
            return out or ["Todo list cleared"], False
        limit = 14 if name in ("task", "delegate", "finish") else 6
        if not lines or r == "(no output)":
            return [gray("(No content)")], False
        out = lines[:limit]
        if len(lines) > limit:
            out.append(gray("… +%d lines" % (len(lines) - limit)))
        return out, False

    # ------------------------------------------------------------------ dialogs
    def confirm(self, title: str, body, question: str, options, danger_note: str = "") -> int:
        """Show a boxed menu. Returns chosen index, or -1 for cancel (Esc/No via Esc)."""
        if not self.interactive:
            return -1
        with self.dialog_lock, self.paused():
            flush_input()
            reader, sel = KeyReader(), 0
            w = min(term_width(), 100)
            drawn = 0

            def draw():
                nonlocal drawn
                lines = [sgr(title, "1;38;5;173"), ""]
                lines += list(body)
                if danger_note:
                    lines += ["", sgr(danger_note, "31")]
                lines += ["", question]
                for i, o in enumerate(options):
                    lines.append((accent(G["arrow"] + " %d. %s" % (i + 1, o)) if i == sel else "  %d. %s" % (i + 1, o)))
                rendered = box(lines, w)
                if drawn:
                    sys.stdout.write("\033[%dA" % drawn)
                sys.stdout.write("\r\033[J" + "\n".join(rendered) + "\n")
                sys.stdout.flush()
                drawn = len(rendered)

            with RawMode():
                sys.stdout.write("\033[?25l")
                try:
                    draw()
                    while True:
                        try:
                            kind, val = reader.read()
                        except EOFError:
                            return -1
                        if kind == "key" and val in ("up", "ctrl-p"):
                            sel = (sel - 1) % len(options)
                        elif kind == "key" and val in ("down", "ctrl-n", "tab"):
                            sel = (sel + 1) % len(options)
                        elif kind == "key" and val == "enter":
                            result = sel
                            break
                        elif kind == "key" and val in ("esc", "ctrl-c"):
                            result = -1
                            break
                        elif kind == "char" and val.isdigit() and 1 <= int(val) <= len(options):
                            result = int(val) - 1
                            break
                        elif kind == "char" and val.lower() in ("y", "n") and len(options) >= 2:
                            result = 0 if val.lower() == "y" else len(options) - 1
                            break
                        draw()
                finally:
                    sys.stdout.write("\033[%dA\r\033[J\033[?25h" % drawn)
                    sys.stdout.flush()
            return result

    # ------------------------------------------------------------------ input box
    def read_input(self, get_mode, cycle_mode, history, commands, right_text=lambda: ""):
        """Boxed editor. commands: list of (name, description). Returns text, or None on exit (Ctrl-D / double Ctrl-C)."""
        if not self.interactive:
            line = sys.stdin.readline()
            return None if line == "" else line.rstrip("\n")
        reader = KeyReader(self._watch.take())
        buf, pos = "", 0
        hist_i, draft = len(history), ""
        sel, drawn_cursor_row, show_help, ctrl_c_t = 0, 0, False, 0.0
        sys.stdout.write("\033[?2004h")

        def wrap(line, tw):
            ranges, s, w = [], 0, 0
            for i, ch in enumerate(line):
                cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
                if w + cw > tw:
                    ranges.append((s, i))
                    s, w = i, 0
                w += cw
            ranges.append((s, len(line)))
            return ranges

        def suggestions():
            if buf.startswith("/") and "\n" not in buf and " " not in buf:
                return [c for c in commands if c[0].startswith(buf)][:7]
            return []

        def render():
            nonlocal drawn_cursor_row
            W = term_width()
            tw = W - 6
            rows, cur, idx = [], (0, 4), 0
            for li, ln in enumerate(buf.split("\n")):
                ranges = wrap(ln, tw)
                off = pos - idx
                for ri, (s, e) in enumerate(ranges):
                    seg = ln[s:e]
                    prefix = accent("> ") if (li == 0 and ri == 0) else "  "
                    rows.append(accent(G["v"]) + " " + prefix + seg + " " * (tw - cell_len(seg)) + " " + accent(G["v"]))
                if 0 <= off <= len(ln):
                    for ri, (s, e) in enumerate(ranges):
                        if off < e or ri == len(ranges) - 1:
                            cur = (1 + len(rows) - len(ranges) + ri, 4 + cell_len(ln[s:off]))
                            break
                idx += len(ln) + 1
            lines = [accent(G["tl"] + G["h"] * (W - 2) + G["tr"])] + rows + [accent(G["bl"] + G["h"] * (W - 2) + G["br"])]
            mode = get_mode()
            hint = {"accept": accent("  %s accept edits on" % G["accept"]) + gray(" (shift+tab to cycle)"),
                    "plan": teal("  %s plan mode on" % G["plan"]) + gray(" (shift+tab to cycle)"),
                    "yolo": sgr("  %s bypass permissions on" % G["accept"], "31") + gray(" (shift+tab to cycle)")}.get(mode, gray("  ? for shortcuts"))
            if ctrl_c_t and time.time() - ctrl_c_t < 2.0 and not buf:
                hint = gray("  Press Ctrl-C again to exit")
            rt = right_text()
            gap = max(1, W - cell_len(hint) - cell_len(rt) - 1)
            lines.append(hint + " " * gap + gray(rt))
            if show_help and not buf:
                for a, b in (("  ! for shell mode", "/ for commands"), ("  esc to interrupt", "shift+tab to cycle modes"),
                             ("  alt+enter or \\+enter", "for newline"), ("  ctrl+c to clear / exit", "tab to complete a command")):
                    lines.append(gray(pad(a, 28) + b))
            sg = suggestions()
            for i, (name, desc) in enumerate(sg):
                row = "  " + pad(name, 18) + desc
                lines.append(accent(clip(row, W)) if i == min(sel, len(sg) - 1) else gray(clip(row, W)))
            out = ""
            if drawn_cursor_row:
                out += "\033[%dA" % drawn_cursor_row
            out += "\r\033[J" + "\n".join(lines)
            up = len(lines) - 1 - cur[0]
            if up:
                out += "\033[%dA" % up
            out += "\r" + ("\033[%dC" % cur[1] if cur[1] else "")
            sys.stdout.write("\033[?25l" + out + "\033[?25h")
            sys.stdout.flush()
            drawn_cursor_row = cur[0]

        def finish():
            if drawn_cursor_row:
                sys.stdout.write("\033[%dA" % drawn_cursor_row)
            sys.stdout.write("\r\033[J\033[?2004l")
            sys.stdout.flush()

        def word_left(p):
            while p > 0 and buf[p - 1].isspace():
                p -= 1
            while p > 0 and not buf[p - 1].isspace():
                p -= 1
            return p

        def word_right(p):
            while p < len(buf) and buf[p].isspace():
                p += 1
            while p < len(buf) and not buf[p].isspace():
                p += 1
            return p

        with RawMode():
            render()
            try:
                while True:
                    try:
                        kind, val = reader.read()
                    except EOFError:
                        finish()
                        return None
                    sg = suggestions()
                    prev_buf = buf
                    if kind == "char":
                        if val == "?" and not buf:
                            show_help = not show_help
                        else:
                            buf, pos = buf[:pos] + val + buf[pos:], pos + len(val)
                            show_help = False
                    elif kind == "paste":
                        buf, pos = buf[:pos] + val + buf[pos:], pos + len(val)
                    elif kind == "key":
                        if val in (("enter",) if IS_WIN else ("enter", "ctrl-j")):
                            if reader.pending():  # pasted text on a terminal without bracketed paste
                                buf, pos = buf[:pos] + "\n" + buf[pos:], pos + 1
                            elif buf.endswith("\\"):
                                buf = buf[:-1] + "\n"
                                pos = len(buf)
                            else:
                                if sg and buf not in [c[0] for c in commands]:
                                    buf = sg[min(sel, len(sg) - 1)][0]
                                finish()
                                return buf
                        elif val in (("alt-enter", "ctrl-j") if IS_WIN else ("alt-enter",)):
                            buf, pos = buf[:pos] + "\n" + buf[pos:], pos + 1
                        elif val == "backspace":
                            if pos:
                                buf, pos = buf[:pos - 1] + buf[pos:], pos - 1
                        elif val == "delete":
                            buf = buf[:pos] + buf[pos + 1:]
                        elif val in ("left", "ctrl-b"):
                            pos = max(0, pos - 1)
                        elif val in ("right", "ctrl-f"):
                            pos = min(len(buf), pos + 1)
                        elif val == "ctrl-left":
                            pos = word_left(pos)
                        elif val == "ctrl-right":
                            pos = word_right(pos)
                        elif val in ("home", "ctrl-a"):
                            pos = buf.rfind("\n", 0, pos) + 1
                        elif val in ("end", "ctrl-e"):
                            nl = buf.find("\n", pos)
                            pos = len(buf) if nl < 0 else nl
                        elif val == "ctrl-u":
                            start = buf.rfind("\n", 0, pos) + 1
                            buf, pos = buf[:start] + buf[pos:], start
                        elif val == "ctrl-k":
                            nl = buf.find("\n", pos)
                            buf = buf[:pos] + ("" if nl < 0 else buf[nl:])
                        elif val == "ctrl-w":
                            p = word_left(pos)
                            buf, pos = buf[:p] + buf[pos:], p
                        elif val in ("up", "ctrl-p"):
                            if sg:
                                sel = (min(sel, len(sg) - 1) - 1) % len(sg)
                            elif "\n" in buf[:pos]:
                                ls = buf.rfind("\n", 0, pos)
                                col = pos - ls - 1
                                ps = buf.rfind("\n", 0, ls) + 1
                                pos = min(ps + col, ls)
                            elif hist_i > 0:
                                if hist_i == len(history):
                                    draft = buf
                                hist_i -= 1
                                buf = history[hist_i]
                                pos = len(buf)
                        elif val in ("down", "ctrl-n"):
                            if sg:
                                sel = (min(sel, len(sg) - 1) + 1) % len(sg)
                            elif "\n" in buf[pos:]:
                                ls = buf.rfind("\n", 0, pos) + 1
                                col = pos - ls
                                nl = buf.find("\n", pos)
                                ne = buf.find("\n", nl + 1)
                                ne = len(buf) if ne < 0 else ne
                                pos = min(nl + 1 + col, ne)
                            elif hist_i < len(history):
                                hist_i += 1
                                buf = draft if hist_i == len(history) else history[hist_i]
                                pos = len(buf)
                        elif val == "tab":
                            if sg:
                                buf = sg[min(sel, len(sg) - 1)][0] + " "
                                pos = len(buf)
                        elif val == "shift-tab":
                            cycle_mode()
                        elif val == "esc":
                            buf, pos, show_help = "", 0, False
                        elif val == "ctrl-l":
                            sys.stdout.write("\033[2J\033[H")
                            drawn_cursor_row = 0
                        elif val == "ctrl-d":
                            if not buf:
                                finish()
                                return None
                        elif val == "ctrl-c":
                            now = time.time()
                            if buf:
                                buf, pos = "", 0
                            elif now - ctrl_c_t < 2.0:
                                finish()
                                return None
                            else:
                                ctrl_c_t = now
                                show_help = False
                    if buf != prev_buf:
                        sel = 0
                    render()
            finally:
                sys.stdout.write("\033[?2004l")
                sys.stdout.flush()


ui = UI()
