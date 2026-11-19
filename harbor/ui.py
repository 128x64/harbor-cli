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

ASCII = util.getenv("ASCII") == "1"      # opt-in only; the default is the full Unicode look
DEFAULT_STYLE = util.getenv("UI", "classic")
if DEFAULT_STYLE not in ("classic", "modern"):
    DEFAULT_STYLE = "classic"
if ASCII:
    G = dict(dot="*", elbow="L", spin=["-", "\\", "|", "/"], tl="+", tr="+", bl="+", br="+", h="-", v="|", arrow=">",
             open="[ ]", done="[x]", prog="[~]", bullet="-", star="*", up="^", down="v", plan="||", accept=">>")
else:
    G = dict(dot="●", elbow="⎿", spin=["·", "✢", "✳", "✶", "✻", "✽"], tl="╭", tr="╮", bl="╰", br="╯", h="─", v="│",
             arrow="❯", open="☐", done="☒", prog="☐", bullet="•", star="✻", up="↑", down="↓", plan="⏸", accept="⏵⏵")

VERBS = ["Thinking", "Pondering", "Cogitating", "Computing", "Working", "Analyzing", "Ruminating", "Reasoning",
         "Brewing", "Synthesizing", "Noodling", "Churning", "Percolating", "Mulling"]
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


def ui_width() -> int:
    """Width shared by the welcome box, input box and dialogs (capped so ultra-wide terminals stay readable)."""
    return min(term_width(), 120)


def box(lines, width, color=accent):
    inner = width - 4
    out = [color(G["tl"] + G["h"] * (width - 2) + G["tr"])]
    for ln in lines:
        out.append(color(G["v"]) + " " + pad(clip(ln, inner), inner) + " " + color(G["v"]))
    out.append(color(G["bl"] + G["h"] * (width - 2) + G["br"]))
    return out


# ---------------------------------------------------------------------------- modes
MODE_STYLE = {   # key -> (border/accent code, badge code)
    "manual": ("38;5;173", "48;5;173;38;5;16;1"),
    "accept": ("38;5;114", "48;5;114;38;5;16;1"),
    "plan": ("38;5;73", "48;5;73;38;5;16;1"),
    "auto": ("38;5;176", "48;5;176;38;5;16;1"),
    "bypass": ("38;5;167", "48;5;167;38;5;231;1"),
    "autobypass": ("38;5;167", "48;5;167;38;5;231;1"),
}


def mode_color(key: str):
    code = MODE_STYLE.get(key, MODE_STYLE["manual"])[0]
    return lambda t: sgr(t, code)


def mode_badge(key: str, label: str) -> str:
    return sgr(" " + label + " ", MODE_STYLE.get(key, MODE_STYLE["manual"])[1])


_FOOT = {   # key -> (icon, text, colour fn name)
    "manual": ("⏵", "manual mode", "gray"), "accept": ("⏵⏵", "accept edits on", "green"), "plan": ("⏸", "plan mode on", "teal"),
    "auto": ("✻", "autonomous mode on", "accent"), "bypass": ("⏵⏵", "bypass permissions on", "red"),
    "autobypass": ("✻⏵", "autonomous + bypass on", "red")}
_FOOT_ASCII = {"manual": ">", "accept": ">>", "plan": "||", "auto": "*", "bypass": ">>", "autobypass": "*>"}


def classic_footer(key: str, short: bool = False) -> str:
    """The Claude Code-style line under the input box. It ALWAYS names the current mode."""
    icon, text, colour = _FOOT.get(key, _FOOT["manual"])
    if ASCII:
        icon = _FOOT_ASCII.get(key, ">")
    paint = {"gray": lambda t: sgr(t, "38;5;250"), "green": lambda t: sgr(t, "38;5;114"), "teal": teal,
             "accent": accent, "red": lambda t: sgr(t, "38;5;167")}[colour]
    head = "  " + paint(icon + " " + text)
    if short:
        return head
    tail = gray(" (shift+tab to cycle)") + (gray("  ·  ? for shortcuts") if key == "manual" else "")
    return head + tail


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


_TBL = (dict(h="-", v="|", tl="+", tm="+", tr="+", ml="+", mm="+", mr="+", bl="+", bm="+", br="+") if ASCII else
        dict(h="─", v="│", tl="┌", tm="┬", tr="┐", ml="├", mm="┼", mr="┤", bl="└", bm="┴", br="┘"))


def render_table(lines, max_width):
    """Markdown pipe table -> boxed, aligned table lines (None if it would not fit)."""
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    is_sep = lambda r: any(r) and all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in r if c)
    has_header = len(rows) > 1 and is_sep(rows[1])
    body = [r for r in rows if not is_sep(r)]
    if not body:
        return None
    n = max(len(r) for r in body)
    body = [r + [""] * (n - len(r)) for r in body]
    cells = [[sgr(re.sub(r"\*\*|`", "", c), "1") if (has_header and ri == 0) else inline_md(c) for c in r] for ri, r in enumerate(body)]
    widths = [max(cell_len(cells[ri][i]) for ri in range(len(body))) for i in range(n)]
    if sum(widths) + 3 * n + 1 > max_width:
        return None
    hline = lambda l, m, r: gray(l + m.join(_TBL["h"] * (w + 2) for w in widths) + r)
    out = [hline(_TBL["tl"], _TBL["tm"], _TBL["tr"])]
    for ri, r in enumerate(cells):
        out.append(gray(_TBL["v"]) + gray(_TBL["v"]).join(" " + c + " " * (widths[i] - cell_len(c)) + " " for i, c in enumerate(r)) + gray(_TBL["v"]))
        if has_header and ri == 0:
            out.append(hline(_TBL["ml"], _TBL["mm"], _TBL["mr"]))
    out.append(hline(_TBL["bl"], _TBL["bm"], _TBL["br"]))
    return out


class MDStream:
    """Feed text chunks; get fully rendered lines as complete lines arrive. Handles headings, lists, quotes, inline code,
    bold/italic, links, fenced code blocks (with a gutter and syntax highlighting) and pipe tables."""

    def __init__(self, style: str = ""):
        self.buf = ""
        self.in_code = False
        self.lang = ""
        self.table = []
        self.modern = (style or DEFAULT_STYLE) == "modern"

    def feed(self, chunk: str):
        self.buf += chunk
        out = []
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            out += self._line(line)
        return out

    def flush(self):
        out = []
        if self.buf:
            line, self.buf = self.buf, ""
            out += self._line(line)
        return out + self._end_table()

    def _end_table(self):
        if not self.table:
            return []
        lines, self.table = self.table, []
        return render_table(lines, term_width() - 6) or [inline_md(l) for l in lines]

    def _line(self, line):
        s = line.strip()
        if self.modern and not self.in_code and s.startswith("|") and s.count("|") >= 2:
            self.table.append(line)
            return []
        out = self._end_table()
        r = self.render(line)
        if r is not None:
            out.append(r)
        return out

    def render(self, line: str):
        line = line.rstrip("\r")
        s = line.strip()
        if s.startswith("```") and not self.modern:      # classic: fences are hidden, code is just indented
            self.in_code = not self.in_code
            self.lang = s[3:].strip().split()[0] if self.in_code and s[3:].strip() else ""
            return None
        if self.in_code and not self.modern:
            return "  " + highlight_code(line, self.lang)
        if s.startswith("```"):
            self.in_code = not self.in_code
            if self.in_code:
                self.lang = s[3:].strip().split()[0] if s[3:].strip() else ""
                return gray(("+- " if ASCII else "┌─ ") + (self.lang or "code"))
            return gray("+-" if ASCII else "└─")
        if self.in_code:
            return gray("|" if ASCII else "│") + " " + highlight_code(line, self.lang)
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            txt = inline_md(m.group(2))
            return accent(sgr(txt, "1;4")) if len(m.group(1)) == 1 else sgr(txt, "1")
        if re.match(r"^\s*([-*_])(\s*\1){2,}\s*$", line):
            return gray(G["h"] * 40)
        m = re.match(r"^(\s*)[-*+]\s+(.*)$", line)
        if m:
            return m.group(1) + G["bullet"] + " " + inline_md(m.group(2))
        if s.startswith(">"):
            return gray("| " if ASCII else "▎ ") + sgr(inline_md(s.lstrip("> ")), "3")
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
        self.quiet = False                         # True while emitting machine-readable output (--output-format json)
        self.style = DEFAULT_STYLE                 # "classic" (the original look) or "modern"
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
        self._spin_chars = 0
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
        if self.quiet:
            return
        with self.lock:
            out = "\r\033[K" if self._spin_vis else ""          # clear the spinner line, print, redraw the spinner:
            self._spin_vis = False                                # ONE write, so terminals never show a half-drawn state
            out += text
            if self._spin_on and not self._spin_paused:
                s = self._spinner_str()
                out += s
                self._spin_vis = bool(s)
            sys.stdout.write(out)
            sys.stdout.flush()
            if text:
                self._last_blank = text.endswith("\n\n") or text.strip() == ""

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
    def _spinner_str(self) -> str:
        if not self.interactive:
            return ""
        el = int(time.time() - self._spin_t0)
        glyph = G["spin"][self._spin_i % len(G["spin"])]
        info = "%ds" % el
        if self._spin_tokens:
            info += " · %s %s tokens" % (G["down"], ("%.1fk" % (self._spin_tokens / 1000)) if self._spin_tokens >= 1000 else self._spin_tokens)
        txt = "%s %s… (%s · esc to interrupt)" % (glyph, self._spin_label, info)
        return "\r\033[K" + accent(clip(txt, term_width()))

    def _draw_spinner(self):
        s = self._spinner_str()
        if s:
            sys.stdout.write(s)
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
        self._spin_t0, self._spin_tokens, self._spin_chars, self._spin_paused = time.time(), 0, 0, False
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
        """Called for every streamed chunk (text, hidden reasoning, tool-call arguments); ~3.8 characters per token."""
        with self.lock:
            self._spin_chars += n_chars
            self._spin_tokens = int(round(self._spin_chars / 3.8))

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
    def set_title(self, text: str):
        if self.interactive:
            sys.stdout.write("\033]0;%s\007" % text.replace("\033", ""))
            sys.stdout.flush()

    def bell(self):
        if self.interactive:
            sys.stdout.write("\a")
            sys.stdout.flush()

    def welcome(self, version, rows, mode_key="manual"):
        """rows: [(label, value)]."""
        if self.style == "modern":
            w = ui_width()
            lines = [accent(G["star"]) + " " + sgr("Harbor CLI", "1") + gray("  v" + version + "  -  agentic coding assistant"), ""]
            lw = max(len(r[0]) for r in rows) if rows else 0
            for k, v in rows:
                lines.append("  " + gray(k.ljust(lw)) + "  " + v)
            for ln in box(lines, w, mode_color(mode_key)):
                self.line(ln)
            self.line()
            self.line(gray(clip(" Ask for anything: Harbor CLI reads, edits, runs and verifies on its own.", w)))
            self.line(gray(clip(" /help commands · @ file paths · shift+tab switches mode · esc interrupts · /auto <goal>", w)))
            self.line()
            return
        w = min(term_width(), 76)
        d = dict(rows)
        lines = [accent(G["star"]) + " " + sgr("Welcome to Harbor CLI!", "1") + gray("  v" + version), "",
                 "  " + gray("/help for help, /status for your current setup"), "",
                 "  " + gray("cwd: ") + d.get("folder", ""), "  " + gray("model: ") + d.get("model", "")]
        for k, v in rows:
            if k not in ("folder", "model"):
                lines.append("  " + gray("%s: %s" % (k, v)))
        for ln in box(lines, w):
            self.line(ln)
        self.line()
        self.line(gray(" Tips for getting started:"))
        for i, t in enumerate(["Ask questions, edit files, or run commands.", "Be specific for the best results.",
                               "Create HARBOR.md files to customize your interaction.",
                               "Shift+Tab cycles modes; Esc interrupts; /auto <goal> works autonomously; /help lists commands."], 1):
            self.line(gray(clip(" %d. %s" % (i, t), term_width())))
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
        if self.md is None:
            self.md, self._first_line = MDStream(self.style), True
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
        if not self.interactive or self.quiet:
            return -1
        with self.dialog_lock, self.paused():
            flush_input()
            reader, sel = KeyReader(), 0
            w = min(ui_width(), 100)
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
    PLACEHOLDER = 'Try "explain this project", "fix the failing tests" ...'

    def read_input(self, get_status, cycle, history, commands, right_text=lambda: "", file_lister=None):
        """Boxed editor with a live mode display. get_status() -> (mode_key, label, description); cycle() switches mode;
        commands: [(name, description)]; file_lister(prefix) -> [relative paths] enables @file completion.
        Returns the text, or None on exit (Ctrl-D / double Ctrl-C)."""
        if not self.interactive:
            line = sys.stdin.readline()
            return None if line == "" else line.rstrip("\n")
        reader = KeyReader(self._watch.take())
        buf, pos = "", 0
        hist_i, draft = len(history), ""
        sel, drawn_cursor_row, show_help, ctrl_c_t = 0, 0, False, 0.0
        last_w = term_width()
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
                return [("cmd", n, n + " ", d, None) for n, d in commands if n.startswith(buf)][:7]
            if file_lister:
                start = max(buf.rfind(" ", 0, pos), buf.rfind("\n", 0, pos)) + 1
                tok = buf[start:pos]
                if tok.startswith("@"):
                    try:
                        files = file_lister(tok[1:])
                    except Exception:
                        files = []
                    return [("file", "@" + f, "@" + f + " ", "", (start, pos)) for f in files[:7]]
            return []

        def apply(item):
            nonlocal buf, pos
            span, repl = item[4], item[2]
            if span:
                buf, pos = buf[:span[0]] + repl + buf[span[1]:], span[0] + len(repl)
            else:
                buf, pos = repl, len(repl)

        def render():
            nonlocal drawn_cursor_row, last_w
            W, last_w = ui_width(), term_width()
            tw = W - 6
            key, label, desc = get_status()
            modern = self.style == "modern"
            col = mode_color(key) if modern else accent
            if not modern:
                W = min(term_width(), 200)
                tw = W - 6
            rows, cur, idx = [], (1, 4), 0
            for li, ln in enumerate(buf.split("\n")):
                ranges = wrap(ln, tw)
                off = pos - idx
                for ri, (s, e) in enumerate(ranges):
                    seg = ln[s:e]
                    prefix = col("> ") if (li == 0 and ri == 0) else "  "
                    body = pad(seg, tw)
                    if modern and not buf and ri == 0:
                        body = pad(gray(clip(self.PLACEHOLDER, tw)), tw)
                    rows.append(col(G["v"]) + " " + prefix + body + " " + col(G["v"]))
                if 0 <= off <= len(ln):
                    for ri, (s, e) in enumerate(ranges):
                        if off < e or ri == len(ranges) - 1:
                            cur = (1 + len(rows) - len(ranges) + ri, 4 + cell_len(ln[s:off]))
                            break
                idx += len(ln) + 1
            if modern:
                lab = " " + label + " "
                top = col(G["tl"] + G["h"]) + sgr(lab, MODE_STYLE.get(key, MODE_STYLE["manual"])[0] + ";1") + col(G["h"] * max(0, W - 3 - cell_len(lab)) + G["tr"])
            else:
                top = col(G["tl"] + G["h"] * (W - 2) + G["tr"])
            lines = [top] + rows + [col(G["bl"] + G["h"] * (W - 2) + G["br"])]
            rt = right_text()
            fits = lambda l, r: cell_len(l) + cell_len(r) + 3 <= W
            if modern:
                badge = mode_badge(key, label)
                left = badge + " " + gray(desc) + gray("  ·  shift+tab to switch")
                if not fits(left, rt):
                    left = badge + " " + gray(desc)
            else:
                left = classic_footer(key)
                if not fits(left, rt):
                    left = classic_footer(key, short=True)
            if not fits(left, rt):
                rt = rt.split(" · ")[-1]              # keep only "NN% ctx"
            if not fits(left, rt):
                rt = ""
            if not fits(left, rt):
                left = clip(left, max(8, W - 6))
            if ctrl_c_t and time.time() - ctrl_c_t < 2.0 and not buf:
                left = gray(" Press Ctrl-C again to exit")
            lines.append((" " if modern else "") + left + " " * max(1, W - cell_len(left) - cell_len(rt) - (2 if modern else 1)) + gray(rt))
            if show_help and not buf:
                for a, b in (("/  commands", "@  file paths (Tab completes)"), ("!  run a shell command", "?  toggle this help"),
                             ("shift+tab / ctrl+t  switch mode", "esc  interrupt - or clear the input"),
                             ("alt+enter or \\+enter  new line", "ctrl+c  clear - twice to exit"),
                             ("modes: manual > accept edits > plan > autonomous", "up/down  history")):
                    lines.append("  " + gray(pad(a, 52) + b))
            sg = suggestions()
            for i, item in enumerate(sg):
                row = "  " + pad(item[1], 22) + item[3]
                lines.append(col(clip(row, W)) if i == min(sel, len(sg) - 1) else gray(clip(row, W)))
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
                        ev = reader.read(poll=0.3)
                    except EOFError:
                        finish()
                        return None
                    if ev is None:                       # idle: redraw if the terminal was resized
                        if term_width() != last_w:
                            render()
                        continue
                    kind, val = ev
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
                            item = sg[min(sel, len(sg) - 1)] if sg else None
                            if reader.pending():                   # pasted text on a terminal without bracketed paste
                                buf, pos = buf[:pos] + "\n" + buf[pos:], pos + 1
                            elif item and item[0] == "file":
                                apply(item)
                            elif buf.endswith("\\"):
                                buf = buf[:-1] + "\n"
                                pos = len(buf)
                            else:
                                if item and item[0] == "cmd" and buf not in [c[0] for c in commands]:
                                    buf = item[1]
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
                                col_ = pos - ls - 1
                                ps = buf.rfind("\n", 0, ls) + 1
                                pos = min(ps + col_, ls)
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
                                col_ = pos - ls
                                nl = buf.find("\n", pos)
                                ne = buf.find("\n", nl + 1)
                                ne = len(buf) if ne < 0 else ne
                                pos = min(nl + 1 + col_, ne)
                            elif hist_i < len(history):
                                hist_i += 1
                                buf = draft if hist_i == len(history) else history[hist_i]
                                pos = len(buf)
                        elif val == "tab":
                            if sg:
                                apply(sg[min(sel, len(sg) - 1)])
                        elif val in ("shift-tab", "ctrl-t"):
                            cycle()
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
