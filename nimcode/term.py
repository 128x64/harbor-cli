"""Raw keyboard handling for POSIX (termios) and Windows (msvcrt), plus the Esc-to-interrupt watcher."""
from __future__ import annotations

import os
import sys
import threading
import time
import _thread

from .util import IS_WIN

if IS_WIN:
    import msvcrt
else:
    try:
        import select
        import termios
    except ImportError:  # pragma: no cover
        termios = None


class RawMode:
    """Put stdin in raw-ish mode (no echo/line buffering/signals; output processing stays on)."""

    def __enter__(self):
        self.old = None
        if not IS_WIN and termios and sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            new = termios.tcgetattr(self.fd)
            new[0] &= ~(termios.ICRNL | termios.INLCR | termios.IXON)
            new[3] &= ~(termios.ECHO | termios.ICANON | termios.ISIG | termios.IEXTEN)
            new[6][termios.VMIN], new[6][termios.VTIME] = 1, 0
            termios.tcsetattr(self.fd, termios.TCSADRAIN, new)
        return self

    def __exit__(self, *exc):
        if self.old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


_CTRL = {1: "ctrl-a", 2: "ctrl-b", 3: "ctrl-c", 4: "ctrl-d", 5: "ctrl-e", 6: "ctrl-f", 8: "backspace", 9: "tab",
         10: "ctrl-j", 11: "ctrl-k", 12: "ctrl-l", 13: "enter", 14: "ctrl-n", 16: "ctrl-p", 18: "ctrl-r",
         21: "ctrl-u", 23: "ctrl-w", 127: "backspace"}
_CSI_FINAL = {"A": "up", "B": "down", "C": "right", "D": "left", "H": "home", "F": "end", "Z": "shift-tab"}
_CSI_TILDE = {"1": "home", "7": "home", "4": "end", "8": "end", "3": "delete", "5": "pgup", "6": "pgdn"}


class KeyReader:
    """read() -> ("char", text) | ("key", name) | ("paste", text). Blocks. Raises EOFError at end of input."""

    def __init__(self, initial=None):
        self.buf = bytearray()
        self.queue = []  # Windows: pre-decoded type-ahead events
        self.fd = sys.stdin.fileno() if not IS_WIN else None
        if initial:
            if IS_WIN:
                self.queue = [("key", "enter") if c in "\r\n" else ("char", c) for c in initial if c in "\r\n" or c >= " "]
            else:
                self.buf += initial

    # ---- POSIX
    def _fill(self, timeout):
        if self.buf:
            return True
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return False
        data = os.read(self.fd, 4096)
        if not data:
            raise EOFError
        self.buf += data
        return True

    def pending(self) -> bool:
        if IS_WIN:
            return bool(self.queue) or msvcrt.kbhit()
        return bool(self.buf) or bool(select.select([self.fd], [], [], 0)[0])

    def read(self):
        return self._read_win() if IS_WIN else self._read_posix()

    def _read_posix(self):
        self._fill(None)
        b = self.buf.pop(0)
        if b == 0x1B:
            if not self._fill(0.03):
                return ("key", "esc")
            nxt = self.buf[0]
            if nxt == 0x1B:          # ESC ESC: the first one is a lone Esc, leave the second for the next read
                return ("key", "esc")
            if nxt in (0x0D, 0x0A):
                self.buf.pop(0)
                return ("key", "alt-enter")
            if nxt == ord("["):
                self.buf.pop(0)
                seq = ""
                while self._fill(0.2):
                    ch = self.buf.pop(0)
                    seq += chr(ch)
                    if 0x40 <= ch <= 0x7E:
                        break
                return self._csi(seq)
            if nxt == ord("O"):
                self.buf.pop(0)
                if self._fill(0.05):
                    return ("key", _CSI_FINAL.get(chr(self.buf.pop(0)), "unknown"))
                return ("key", "unknown")
            self.buf.pop(0)
            return ("key", "alt-" + chr(nxt))
        if b < 0x20 or b == 0x7F:
            return ("key", _CTRL.get(b, "ctrl-?"))
        n = 1 if b < 0x80 else 2 if b >> 5 == 0b110 else 3 if b >> 4 == 0b1110 else 4
        raw = bytearray([b])
        while len(raw) < n and self._fill(0.2):
            raw.append(self.buf.pop(0))
        return ("char", raw.decode("utf-8", "replace"))

    def _csi(self, seq):
        if not seq:
            return ("key", "unknown")
        final, params = seq[-1], seq[:-1]
        if final == "~":
            num = params.split(";")[0]
            if num == "200":  # bracketed paste
                data = bytearray()
                end = b"\x1b[201~"
                while True:
                    if not self._fill(1.0):
                        break
                    data.append(self.buf.pop(0))
                    if data.endswith(end):
                        del data[-len(end):]
                        break
                return ("paste", data.decode("utf-8", "replace").replace("\r\n", "\n").replace("\r", "\n"))
            return ("key", _CSI_TILDE.get(num, "unknown"))
        name = _CSI_FINAL.get(final, "unknown")
        if ";5" in params and name in ("left", "right"):
            return ("key", "ctrl-" + name)
        return ("key", name)

    # ---- Windows
    def _read_win(self):
        if self.queue:
            return self.queue.pop(0)
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            c2 = msvcrt.getwch()
            names = {"H": "up", "P": "down", "K": "left", "M": "right", "G": "home", "O": "end", "S": "delete",
                     "s": "ctrl-left", "t": "ctrl-right", "\x0f": "shift-tab"}
            return ("key", names.get(c2, "unknown"))
        if ch == "\r":
            return ("key", "enter")
        if ch == "\x1b":
            return ("key", "esc")
        if ch == "\x7f":
            return ("key", "ctrl-w")
        if ord(ch) < 0x20:
            return ("key", _CTRL.get(ord(ch), "ctrl-?"))
        return ("char", ch)


class EscWatcher:
    """While running, a lone Esc key press interrupts the main thread (like Claude Code's 'esc to interrupt')."""

    def __init__(self):
        self.thread = None
        self.stop_evt = threading.Event()
        self.typeahead = bytearray()   # POSIX raw bytes typed while the agent was working
        self.win_typeahead = ""        # Windows characters

    def take(self):
        """Return (and clear) what the user typed while the agent was busy."""
        if IS_WIN:
            s, self.win_typeahead = self.win_typeahead, ""
            return s
        b, self.typeahead = bytes(self.typeahead), bytearray()
        return b

    def start(self):
        if self.thread or not (sys.stdin.isatty() and sys.stdout.isatty()):
            return
        self.stop_evt.clear()
        self.thread = threading.Thread(target=self._run_win if IS_WIN else self._run_posix, daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread:
            self.stop_evt.set()
            self.thread.join(1.5)
            self.thread = None

    def _run_posix(self):
        if not termios:
            return
        fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd)
            new = termios.tcgetattr(fd)
            new[0] &= ~(termios.ICRNL | termios.INLCR)  # keep Enter as \r so type-ahead is replayed faithfully
            new[3] &= ~(termios.ECHO | termios.ICANON)  # ISIG stays on so Ctrl-C still works
            new[6][termios.VMIN], new[6][termios.VTIME] = 1, 0
            termios.tcsetattr(fd, termios.TCSANOW, new)
        except Exception:
            return
        try:
            while not self.stop_evt.is_set():
                if select.select([fd], [], [], 0.1)[0]:
                    data = os.read(fd, 4096)
                    if data == b"\x1b":
                        _thread.interrupt_main()
                        break
                    self.typeahead += data
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSANOW, old)
            except Exception:
                pass

    def _run_win(self):
        while not self.stop_evt.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):
                    msvcrt.getwch()
                elif ch == "\x1b":
                    _thread.interrupt_main()
                    break
                else:
                    self.win_typeahead += ch
            else:
                time.sleep(0.05)


def flush_input() -> None:
    """Discard pending keystrokes (so stray type-ahead can never answer a permission dialog)."""
    try:
        if IS_WIN:
            while msvcrt.kbhit():
                msvcrt.getwch()
        elif termios and sys.stdin.isatty():
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass
