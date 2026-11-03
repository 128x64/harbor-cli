"""Terminal, text and platform helpers (Windows / macOS / Linux)."""
from __future__ import annotations

import locale
import os
import sys
from pathlib import Path

IS_WIN = os.name == "nt"
USE_COLOR = False


def init_terminal() -> None:
    """Enable UTF-8 output and ANSI colors (including legacy Windows consoles)."""
    global USE_COLOR
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if IS_WIN:
        try:
            import ctypes
            k = ctypes.windll.kernel32
            for std in (-11, -12):
                h = k.GetStdHandle(std)
                mode = ctypes.c_uint32()
                if k.GetConsoleMode(h, ctypes.byref(mode)):
                    k.SetConsoleMode(h, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        except Exception:
            pass
    USE_COLOR = bool(sys.stdout.isatty()) and not os.environ.get("NO_COLOR")


def _c(text: str, code: str) -> str:
    return "\033[%sm%s\033[0m" % (code, text) if USE_COLOR else text


def dim(t): return _c(t, "2")
def bold(t): return _c(t, "1")
def red(t): return _c(t, "31")
def green(t): return _c(t, "32")
def yellow(t): return _c(t, "33")
def cyan(t): return _c(t, "36")


def truncate(text: str, limit: int = 20000) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n... [truncated %d chars] ...\n" % (len(text) - limit) + text[-half:]


def truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


def color_diff(lines) -> str:
    out = []
    for line in lines:
        line = line.rstrip("\n")
        if line.startswith("+") and not line.startswith("+++"):
            out.append(green(line))
        elif line.startswith("-") and not line.startswith("---"):
            out.append(red(line))
        elif line.startswith("@@"):
            out.append(cyan(line))
        else:
            out.append(dim(line))
    return "\n".join(out)


def decode_bytes(b: bytes) -> str:
    """Decode process output: UTF-8 first, then the console/locale code page."""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode(locale.getpreferredencoding(False) or "utf-8", "replace")


def read_text_file(path: Path):
    """Return (text with LF newlines, original newline style, had_bom). Preserves CRLF files."""
    raw = Path(path).read_bytes()
    bom = raw.startswith(b"\xef\xbb\xbf")
    if bom:
        raw = raw[3:]
    text = raw.decode("utf-8", errors="replace")
    nl = "\r\n" if "\r\n" in text else "\n"
    return text.replace("\r\n", "\n"), nl, bom


def write_text_file(path: Path, text: str, nl: str = "\n", bom: bool = False) -> None:
    data = text.replace("\n", nl) if nl != "\n" else text
    raw = data.encode("utf-8")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes((b"\xef\xbb\xbf" if bom else b"") + raw)


def have(module: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False
