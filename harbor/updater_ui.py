"""Installer window for Harbor CLI updates: Tkinter, with a console fallback (no Tk / no display / HARBOR_UPDATE_CONSOLE=1).

Run by harbor.updater as a separate process:  python -m harbor.updater_ui <jobfile>
Exit status: 0 updated, 10 not now, 11 skip this version, 20 failed. A result JSON is written next to the job file."""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

from .updater import EXIT, UpdateError, UpdateJob


def run_job(job: dict, progress) -> dict:
    """Run the update and ALWAYS return a result dict (never raises)."""
    try:
        return UpdateJob(job["info"], Path(job["root"]), progress, deps=job.get("deps", True), python=job.get("python")).run()
    except UpdateError as e:
        return {"status": "failed", "error": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "error": "%s: %s" % (type(e).__name__, e)}


def countdown_seconds(default: int) -> int:
    try:
        return max(0, int(os.environ.get("HARBOR_UPDATE_COUNTDOWN", default)))
    except ValueError:
        return default


# --------------------------------------------------------------------------------------------- console installer
class ConsoleView:
    def __init__(self, job: dict, inp=input, out=print, sleep=time.sleep):
        self.job, self.inp, self.out, self.sleep = job, inp, out, sleep

    def run(self) -> dict:
        info, mode = self.job["info"], self.job.get("mode", "auto")
        self.out("")
        self.out("=== Harbor CLI updater: %s -> %s ===" % (self.job.get("from", "?"), info["version"]))
        for line in (info.get("notes") or "").strip().splitlines()[:12]:
            self.out("  " + line)
        try:
            if mode == "ask":
                ans = self.inp("Install now? [Y]es / [n]ot now / [s]kip this version: ").strip().lower()
                if ans.startswith("n"):
                    return {"status": "later"}
                if ans.startswith("s"):
                    return {"status": "skipped"}
            else:
                for left in range(countdown_seconds(3), 0, -1):
                    self.out("Updating automatically in %d s ... (Ctrl+C to cancel)" % left)
                    self.sleep(1)
        except (KeyboardInterrupt, EOFError):
            return {"status": "later"}
        result = run_job(self.job, lambda msg, pct: self.out("[%3d%%] %s" % (pct, msg)))
        self.out("Done. Restarting Harbor CLI ..." if result["status"] == "updated" else "Update failed: %s" % result.get("error"))
        return result


# --------------------------------------------------------------------------------------------- Tk installer window
class TkView:
    def __init__(self, tk, ttk, job: dict):
        self.tk, self.ttk, self.job = tk, ttk, job
        self.info, self.mode = job["info"], job.get("mode", "auto")
        self.q: queue.Queue = queue.Queue()
        self.result: dict = {"status": "later"}
        self.installing = self.finished = False
        self.started = False
        self.left = countdown_seconds(5)

    def build(self):
        tk, ttk, info = self.tk, self.ttk, self.info
        root = self.root = tk.Tk()
        root.title("Harbor CLI Updater")
        root.geometry("580x480")
        root.resizable(False, False)
        try:
            w, h = root.winfo_screenwidth(), root.winfo_screenheight()
            root.geometry("+%d+%d" % ((w - 580) // 2, (h - 480) // 3))
            root.attributes("-topmost", True)
            root.after(1500, lambda: root.attributes("-topmost", False))
        except Exception:
            pass
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        ttk.Label(root, text="Harbor CLI %s is available" % info["version"], font=("TkDefaultFont", 15, "bold")).pack(anchor="w", padx=16, pady=(16, 0))
        ttk.Label(root, text="You have %s.  Harbor CLI stops, installs the update, and starts again." % self.job.get("from", "?")).pack(anchor="w", padx=16, pady=(2, 8))
        frame = ttk.Frame(root)
        frame.pack(fill="both", expand=True, padx=16)
        self.notes = tk.Text(frame, height=11, wrap="word")
        scroll = ttk.Scrollbar(frame, command=self.notes.yview)
        self.notes.configure(yscrollcommand=scroll.set)
        self.notes.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.notes.insert("end", (info.get("notes") or "").strip() or "No release notes were published for this version.")
        self.notes.configure(state="disabled")
        self.deps_var = tk.IntVar(value=1 if self.job.get("deps", True) else 0)
        ttk.Checkbutton(root, text="Install updated Python dependencies (pip)", variable=self.deps_var).pack(anchor="w", padx=16, pady=(8, 0))
        self.bar = ttk.Progressbar(root, maximum=100, mode="determinate")
        self.bar.pack(fill="x", padx=16, pady=(8, 2))
        self.status = tk.StringVar(value="")
        ttk.Label(root, textvariable=self.status, wraplength=540).pack(anchor="w", padx=16)
        buttons = ttk.Frame(root)
        buttons.pack(fill="x", padx=16, pady=14)
        self.btn_install = ttk.Button(buttons, text="Install now", command=self.start_install)
        self.btn_later = ttk.Button(buttons, text="Not now", command=self.later)
        self.btn_skip = ttk.Button(buttons, text="Skip this version", command=self.skip)
        self.btn_close = ttk.Button(buttons, text="Close", command=self.close_after_failure)
        for b in (self.btn_install, self.btn_later, self.btn_skip):
            b.pack(side="left", padx=(0, 8))

    # ---- actions
    def later(self):
        if self.installing:
            return
        self.result = {"status": "later"}
        self.root.destroy()

    def skip(self):
        if self.installing:
            return
        self.result = {"status": "skipped"}
        self.root.destroy()

    def on_close(self):
        if not self.installing:
            self.later()

    def close_after_failure(self):
        self.root.destroy()

    def tick(self):
        """Automatic mode: count down, then install (the user can still click Not now / Skip)."""
        if self.installing or self.finished:
            return
        if self.left <= 0:
            self.start_install()
            return
        self.status.set("Installing automatically in %d s - click 'Not now' to cancel." % self.left)
        self.left -= 1
        self.root.after(1000, self.tick)

    def start_install(self):
        if self.installing:
            return
        self.installing = self.started = True
        for b in (self.btn_install, self.btn_later, self.btn_skip):
            b.configure(state="disabled")
        job = dict(self.job, deps=bool(self.deps_var.get()))
        self.status.set("Starting ...")

        def work():
            res = run_job(job, lambda msg, pct: self.q.put(("p", msg, pct)))
            self.q.put(("done", res))
        threading.Thread(target=work, daemon=True).start()
        self.root.after(100, self.poll)

    def poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] == "p":
                    self.status.set(item[1])
                    self.bar.configure(value=item[2])
                else:
                    self.finish(item[1])
                    return
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def finish(self, res: dict):
        self.finished = True
        self.installing = False
        self.result = res
        if res.get("status") == "updated":
            self.bar.configure(value=100)
            self.status.set("Updated to %s. Restarting Harbor CLI ..." % res.get("to", ""))
            self.root.after(1400, self.root.destroy)
        else:
            self.status.set("Update failed: %s" % res.get("error", "unknown error"))
            self.btn_close.pack(side="left")

    def run(self) -> dict:
        self.build()
        if self.mode != "ask":
            self.root.after(300, self.tick)
        self.root.mainloop()
        return self.result


def main(argv=None) -> int:
    argv = argv or sys.argv
    jobfile = argv[1]
    job = json.loads(Path(jobfile).read_text(encoding="utf-8"))
    result = None
    if not os.environ.get("HARBOR_UPDATE_CONSOLE"):
        view = None
        try:
            import tkinter as tk
            from tkinter import ttk
            view = TkView(tk, ttk, job)
            result = view.run()
        except Exception as e:  # noqa: BLE001 - no tkinter / no display / window failure
            if view is not None and view.started:
                result = {"status": "failed", "error": "installer window failed: %s" % e}
            else:
                result = None                       # nothing was started: fall back to the console installer
    if result is None:
        result = ConsoleView(job).run()
    try:
        Path(jobfile + ".result").write_text(json.dumps(result), encoding="utf-8")
    except OSError:
        pass
    return EXIT.get(result.get("status"), 20)


if __name__ == "__main__":
    sys.exit(main())
