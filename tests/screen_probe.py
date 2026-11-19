"""Drive Harbor CLI in a pty and print the emulated screen. Usage: python screen_probe.py '<keys>' ...  (keys use \\r \\x1b[Z escapes)"""
import os, select, struct, subprocess, sys, tempfile, time, threading, fcntl, termios, pty
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import codecs
import mock_nim
from vt import Screen


class Probe:
    def __init__(self, extra_args=(), rows=36, cols=100, env=None, port=None):
        self.srv = mock_nim.serve(0); self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.home, self.cwd = tempfile.mkdtemp(), tempfile.mkdtemp()
        self.master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        e = dict(os.environ, NVIDIA_API_KEY="nvapi-test", HOME=self.home, PYTHONPATH=ROOT, TERM="xterm-256color"); e.update(env or {})
        e.pop("NO_COLOR", None)
        self.p = subprocess.Popen([sys.executable, "-m", "harbor", "--base-url", "http://127.0.0.1:%d/v1" % self.port, "--cwd", self.cwd] + list(extra_args),
                                  stdin=slave, stdout=slave, stderr=slave, env=e, cwd=self.cwd, close_fds=True, start_new_session=True)
        os.close(slave)
        self.screen = Screen(rows, cols); self.raw = ""; self._dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def pump(self, secs=0.8):
        end = time.time() + secs
        while time.time() < end:
            r, _, _ = select.select([self.master], [], [], 0.1)
            if r:
                try: d = self._dec.decode(os.read(self.master, 65536))
                except OSError: break
                self.raw += d; self.screen.feed(d)

    def send(self, s, wait=0.6):
        os.write(self.master, s.encode()); self.pump(wait)

    def show(self, title=""):
        print("=" * 30, title); print(self.screen.text(include_scrollback=True)); print("=" * 30)

    def close(self):
        try: self.p.kill()
        except Exception: pass
        os.close(self.master); self.srv.shutdown()


if __name__ == "__main__":
    p = Probe(sys.argv[1:] if sys.argv[1:] else [])
    p.pump(1.5); p.show("startup")
    p.send("\x1b[Z"); p.show("after shift+tab")
    p.send("\x1b[Z"); p.send("\x1b[Z"); p.show("after 3x shift+tab")
    p.close()
