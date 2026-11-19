"""A tiny VT100 screen emulator (enough for Harbor CLI's UI) so tests can look at the *rendered* screen."""
import re
import unicodedata

_CSI = re.compile(r"\x1b\[([0-9;?]*)([ -/]*)([@-~])")


class Screen:
    def __init__(self, rows=40, cols=100):
        self.rows, self.cols = rows, cols
        self.grid = [[" "] * cols for _ in range(rows)]
        self.scrollback = []
        self.r = self.c = 0
        self.pending_wrap = False
        self.saved = (0, 0)

    def _scroll(self):
        self.scrollback.append("".join(self.grid.pop(0)).rstrip())
        self.grid.append([" "] * self.cols)

    def _newline(self):
        if self.r == self.rows - 1:
            self._scroll()
        else:
            self.r += 1

    def _put(self, ch):
        w = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else (0 if unicodedata.combining(ch) else 1)
        if w == 0:
            return
        if self.pending_wrap or self.c + w > self.cols:
            self.c = 0
            self._newline()
            self.pending_wrap = False
        self.grid[self.r][self.c] = ch
        if w == 2 and self.c + 1 < self.cols:
            self.grid[self.r][self.c + 1] = ""
        self.c += w
        if self.c >= self.cols:
            self.c = self.cols - 1
            self.pending_wrap = True

    def feed(self, data: str):
        data = getattr(self, "_carry", "") + data
        self._carry = ""
        k = data.rfind("\x1b")
        if k >= 0:                                   # keep an incomplete trailing escape sequence for the next chunk
            tail = data[k:]
            complete = (_CSI.match(tail) is not None or (tail.startswith("\x1b]") and ("\x07" in tail or "\x1b\\" in tail))
                        or (len(tail) >= 2 and tail[1] not in "[]"))
            if not complete:
                self._carry, data = tail, data[:k]
        i, n = 0, len(data)
        while i < n:
            ch = data[i]
            if ch == "\x1b":
                m = _CSI.match(data, i)
                if m:
                    self._csi(m.group(1), m.group(3))
                    i = m.end()
                    continue
                if data.startswith("\x1b]", i):          # OSC ... BEL / ST
                    j = min([k for k in (data.find("\x07", i), data.find("\x1b\\", i)) if k >= 0] or [n - 1])
                    i = j + 1
                    continue
                if data.startswith("\x1b7", i):
                    self.saved = (self.r, self.c)
                    i += 2
                    continue
                if data.startswith("\x1b8", i):
                    self.r, self.c = self.saved
                    i += 2
                    continue
                i += 2
                continue
            if ch == "\r":
                self.c, self.pending_wrap = 0, False
            elif ch == "\n":
                self.pending_wrap = False
                self._newline()
            elif ch == "\b":
                self.c = max(0, self.c - 1)
            elif ch == "\x07" or ord(ch) < 32:
                pass
            else:
                self._put(ch)
            i += 1

    def _csi(self, params, final):
        priv = params.startswith("?")
        nums = [int(x) if x.isdigit() else 0 for x in params.lstrip("?").split(";")] if params.lstrip("?") else []
        n1 = nums[0] if nums and nums[0] else 1
        if priv:
            return
        self.pending_wrap = False
        if final == "A":
            self.r = max(0, self.r - n1)
        elif final == "B":
            self.r = min(self.rows - 1, self.r + n1)
        elif final == "C":
            self.c = min(self.cols - 1, self.c + n1)
        elif final == "D":
            self.c = max(0, self.c - n1)
        elif final == "G":
            self.c = min(self.cols - 1, n1 - 1)
        elif final in "Hf":
            self.r = min(self.rows - 1, (nums[0] or 1) - 1 if nums else 0)
            self.c = min(self.cols - 1, (nums[1] or 1) - 1 if len(nums) > 1 else 0)
        elif final == "J":
            mode = nums[0] if nums else 0
            if mode == 0:
                self.grid[self.r][self.c:] = [" "] * (self.cols - self.c)
                for k in range(self.r + 1, self.rows):
                    self.grid[k] = [" "] * self.cols
            elif mode in (2, 3):
                self.grid = [[" "] * self.cols for _ in range(self.rows)]
        elif final == "K":
            mode = nums[0] if nums else 0
            if mode == 0:
                self.grid[self.r][self.c:] = [" "] * (self.cols - self.c)
            elif mode == 2:
                self.grid[self.r] = [" "] * self.cols

    def lines(self):
        return ["".join(row).rstrip() for row in self.grid]

    def text(self, include_scrollback=False):
        ls = self.lines()
        while ls and not ls[-1]:
            ls.pop()
        return "\n".join((self.scrollback if include_scrollback else []) + ls)
