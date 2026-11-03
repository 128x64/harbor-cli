"""Filesystem tools: read (text/PDF/DOCX/XLSX/PPTX/ipynb), write, edit, search, move, archive."""
from __future__ import annotations

import difflib
import fnmatch
import hashlib
import json
import os
import re
import shutil
import tarfile
import zipfile
from pathlib import Path

from ..core import Registry
from ..util import (bold, color_diff, cyan, dim, green, read_text_file, truncate, truthy, write_text_file)

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", "dist",
             "build", ".idea", ".next", "target", ".tox", ".vscode-test", "$RECYCLE.BIN", "System Volume Information"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}


def glob_to_regex(pat: str):
    pat = pat.replace("\\", "/")
    if "/" not in pat:
        pat = "**/" + pat
    i, out = 0, ""
    while i < len(pat):
        if pat[i:i + 3] == "**/":
            out += "(?:.*/)?"
            i += 3
        elif pat[i:i + 2] == "**":
            out += ".*"
            i += 2
        elif pat[i] == "*":
            out += "[^/]*"
            i += 1
        elif pat[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(pat[i])
            i += 1
    return re.compile("^" + out + "$", re.I if os.name == "nt" else 0)


def walk_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fn in sorted(filenames):
            yield Path(dirpath) / fn


# ---------------------------------------------------------------- document readers
def _read_pdf(p: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "Error: reading PDFs needs `pip install pypdf`."
    r = PdfReader(str(p))
    return "\n".join("--- page %d ---\n%s" % (i, pg.extract_text() or "") for i, pg in enumerate(r.pages, 1))


def _read_docx(p: Path) -> str:
    try:
        import docx
    except ImportError:
        return "Error: reading .docx needs `pip install python-docx`."
    d = docx.Document(str(p))
    lines = [para.text for para in d.paragraphs]
    for i, t in enumerate(d.tables, 1):
        lines.append("[table %d]" % i)
        for row in t.rows:
            lines.append("\t".join(c.text.strip() for c in row.cells))
    return "\n".join(lines)


def _read_xlsx(p: Path) -> str:
    try:
        import openpyxl
    except ImportError:
        return "Error: reading .xlsx needs `pip install openpyxl`."
    wb = openpyxl.load_workbook(str(p), read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        out.append("=== sheet: %s ===" % ws.title)
        for n, row in enumerate(ws.iter_rows(values_only=True), 1):
            if n > 1000:
                out.append("... (more rows)")
                break
            out.append("\t".join("" if c is None else str(c) for c in row))
    wb.close()
    return "\n".join(out)


def _read_pptx(p: Path) -> str:
    try:
        from pptx import Presentation
    except ImportError:
        return "Error: reading .pptx needs `pip install python-pptx`."
    out = []
    for i, slide in enumerate(Presentation(str(p)).slides, 1):
        out.append("--- slide %d ---" % i)
        for sh in slide.shapes:
            if getattr(sh, "has_text_frame", False) and sh.text_frame.text.strip():
                out.append(sh.text_frame.text)
    return "\n".join(out)


def _read_ipynb(p: Path) -> str:
    nb = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    out = []
    for i, c in enumerate(nb.get("cells", []), 1):
        out.append("--- cell %d [%s] ---\n%s" % (i, c.get("cell_type"), "".join(c.get("source", []))))
        for o in c.get("outputs", []) or []:
            txt = "".join(o.get("text", [])) or "".join((o.get("data", {}) or {}).get("text/plain", []))
            if txt:
                out.append("[output]\n" + txt[:2000])
    return "\n".join(out)


READERS = {".pdf": _read_pdf, ".docx": _read_docx, ".xlsx": _read_xlsx, ".xlsm": _read_xlsx,
           ".pptx": _read_pptx, ".ipynb": _read_ipynb}


# ---------------------------------------------------------------- edit helpers
def _closest(text: str, needle: str) -> str:
    first = next((ln for ln in needle.splitlines() if ln.strip()), "")
    if not first:
        return ""
    m = difflib.get_close_matches(first.strip(), [l.strip() for l in text.splitlines()], n=1, cutoff=0.6)
    return "\nClosest line in file: %r" % m[0] if m else ""


def apply_edits(text: str, edits: list):
    """Apply [(old, new, replace_all)] to text. Returns (new_text, error)."""
    for n, (old, new, rep_all) in enumerate(edits, 1):
        old, new = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
        if old == "":
            return None, "edit %d: old_string is empty" % n
        c = text.count(old)
        if c == 0:
            return None, "edit %d: old_string not found. Re-read the file and copy the exact text incl. indentation.%s" % (n, _closest(text, old))
        if c > 1 and not rep_all:
            return None, "edit %d: old_string matches %d places; add surrounding context or set replace_all=true." % (n, c)
        text = text.replace(old, new) if rep_all else text.replace(old, new, 1)
    return text, None


def register(reg: Registry) -> None:
    @reg.tool("read_file",
              "Read a file with line numbers. Handles text/code, PDF, DOCX, XLSX, PPTX and Jupyter notebooks. "
              "Use offset/limit (lines) for big files. For images use analyze_image.",
              {"path": {"type": "string"}, "offset": {"type": "integer", "description": "1-based start line"},
               "limit": {"type": "integer", "description": "max lines (default 2000)"}},
              ["path"], group="core", read_only=True, summary_key="path")
    def read_file(ctx, path, offset=1, limit=2000):
        p = ctx.path(path)
        if not p.exists():
            return "Error: file not found: %s" % p
        if p.is_dir():
            return "Error: %s is a directory; use list_dir." % p
        ext = p.suffix.lower()
        if ext in IMAGE_EXT:
            return "This is an image file. Use analyze_image (vision) or ocr_image to inspect it."
        if ext in READERS:
            try:
                text = READERS[ext](p)
            except Exception as e:
                return "Error reading %s: %s: %s" % (p.name, type(e).__name__, e)
            if text.startswith("Error:"):
                return text
        else:
            raw = p.read_bytes()
            if b"\0" in raw[:8192]:
                return "Error: %s looks like a binary file (%d bytes)." % (p, len(raw))
            text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        offset, limit = max(1, int(offset)), max(1, int(limit))
        chunk = lines[offset - 1: offset - 1 + limit]
        out = "\n".join("%6d\t%s" % (i, ln[:2000]) for i, ln in enumerate(chunk, start=offset))
        rest = len(lines) - (offset - 1 + limit)
        if rest > 0:
            out += "\n... (%d more lines; use offset=%d)" % (rest, offset + limit)
        return out or "(empty file)"

    def _write_preview(ctx, a):
        p = ctx.path(a.get("path", ""))
        new = str(a.get("content", ""))
        head = bold("● write_file ") + cyan(str(p))
        if p.is_file():
            old = read_text_file(p)[0]
            diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), "old", "new", lineterm="", n=1))
            return head + dim(" (overwrite)") + "\n" + color_diff(diff[:80])
        lines = new.splitlines()
        body = "\n".join(green("+ " + ln) for ln in lines[:20])
        more = len(lines) - 20
        return head + dim(" (new file, %d lines)" % len(lines)) + "\n" + body + (dim("\n  ... +%d more lines" % more) if more > 0 else "")

    @reg.tool("write_file", "Create a new file or fully overwrite an existing one (parent dirs are created).",
              {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"],
              group="core", ask=True, preview=_write_preview, summary_key="path")
    def write_file(ctx, path, content):
        p = ctx.path(path)
        if p.is_dir():
            return "Error: %s is a directory." % p
        existed, nl, bom = p.exists(), "\n", False
        if existed:
            try:
                _, nl, bom = read_text_file(p)  # keep CRLF / BOM of existing files
            except OSError:
                pass
        write_text_file(p, content, nl, bom)
        return "%s %s (%d lines)" % ("Overwrote" if existed else "Created", p, len(content.splitlines()))

    def _edit_preview(ctx, a):
        head = bold("● edit_file ") + cyan(str(a.get("path", "")))
        p = ctx.path(a.get("path", ""))
        if not p.is_file():
            return head
        old, _, _ = read_text_file(p)
        new, err = apply_edits(old, [(a.get("old_string", ""), a.get("new_string", ""), truthy(a.get("replace_all", False)))])
        if err:
            return head
        diff = difflib.unified_diff(old.splitlines(), new.splitlines(), "a/" + p.name, "b/" + p.name, lineterm="", n=2)
        return head + "\n" + color_diff(list(diff)[:80])

    @reg.tool("edit_file",
              "Replace an exact string in a file. old_string must match exactly once unless replace_all=true. "
              "To create a new file pass an empty old_string.",
              {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"},
               "replace_all": {"type": "boolean"}}, ["path", "old_string", "new_string"],
              group="core", ask=True, preview=_edit_preview, summary_key="path")
    def edit_file(ctx, path, old_string, new_string, replace_all=False):
        p = ctx.path(path)
        if not p.exists():
            if old_string == "":
                write_text_file(p, new_string)
                return "Created %s" % p
            return "Error: file not found: %s" % p
        text, nl, bom = read_text_file(p)
        new, err = apply_edits(text, [(old_string, new_string, truthy(replace_all))])
        if err:
            return "Error: " + err
        write_text_file(p, new, nl, bom)
        return "Edited %s" % p

    def _multi_preview(ctx, a):
        head = bold("● multi_edit ") + cyan(str(a.get("path", ""))) + dim(" (%d edits)" % len(a.get("edits") or []))
        p = ctx.path(a.get("path", ""))
        if not p.is_file() or not isinstance(a.get("edits"), list):
            return head
        old = read_text_file(p)[0]
        new, err = apply_edits(old, [(e.get("old_string", ""), e.get("new_string", ""), truthy(e.get("replace_all"))) for e in a["edits"] if isinstance(e, dict)])
        if err:
            return head
        return head + "\n" + color_diff(list(difflib.unified_diff(old.splitlines(), new.splitlines(), "a", "b", lineterm="", n=2))[:100])

    @reg.tool("multi_edit", "Apply several edits to ONE file atomically (all or none), in order.",
              {"path": {"type": "string"},
               "edits": {"type": "array", "items": {"type": "object", "properties": {
                   "old_string": {"type": "string"}, "new_string": {"type": "string"},
                   "replace_all": {"type": "boolean"}}, "required": ["old_string", "new_string"]}}},
              ["path", "edits"], group="core", ask=True, preview=_multi_preview, summary_key="path")
    def multi_edit(ctx, path, edits):
        p = ctx.path(path)
        if not p.is_file():
            return "Error: file not found: %s" % p
        if not isinstance(edits, list) or not edits:
            return "Error: edits must be a non-empty list."
        text, nl, bom = read_text_file(p)
        new, err = apply_edits(text, [(e.get("old_string", ""), e.get("new_string", ""), truthy(e.get("replace_all"))) for e in edits])
        if err:
            return "Error (nothing written): " + err
        write_text_file(p, new, nl, bom)
        return "Applied %d edits to %s" % (len(edits), p)

    @reg.tool("list_dir", "List a directory. depth>1 shows a tree (skips .git, node_modules, venvs...).",
              {"path": {"type": "string"}, "depth": {"type": "integer", "description": "default 1"}},
              [], group="core", read_only=True, summary_key="path")
    def list_dir(ctx, path=".", depth=1):
        root = ctx.path(path)
        if not root.is_dir():
            return "Error: not a directory: %s" % root
        depth, lines = max(1, min(int(depth), 6)), []

        def rec(d: Path, level: int):
            try:
                entries = sorted(d.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
            except OSError as e:
                lines.append("  " * level + "[unreadable: %s]" % e)
                return
            for e in entries:
                if len(lines) >= 500:
                    return
                if e.is_dir():
                    lines.append("  " * level + e.name + "/")
                    if level + 1 < depth and e.name not in SKIP_DIRS:
                        rec(e, level + 1)
                else:
                    lines.append("  " * level + e.name)
        rec(root, 0)
        if len(lines) >= 500:
            lines.append("... (truncated)")
        return "\n".join(lines) or "(empty directory)"

    @reg.tool("glob", "Find files by glob (supports ** and *), newest first. e.g. '**/*.py' or '*.md'.",
              {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"],
              group="core", read_only=True, summary_key="pattern")
    def glob(ctx, pattern, path="."):
        root = ctx.path(path)
        rx = glob_to_regex(pattern)
        hits = []
        for f in walk_files(root):
            rel = f.relative_to(root).as_posix()
            if rx.match(rel):
                try:
                    hits.append((f.stat().st_mtime, rel))
                except OSError:
                    pass
        hits.sort(reverse=True)
        if not hits:
            return "No files matched."
        out = [h[1] for h in hits[:200]]
        if len(hits) > 200:
            out.append("... (%d more)" % (len(hits) - 200))
        return "\n".join(out)

    @reg.tool("grep", "Regex search in file contents. Returns path:line:text.",
              {"pattern": {"type": "string"}, "path": {"type": "string"},
               "glob": {"type": "string", "description": "only files matching, e.g. '*.py'"},
               "ignore_case": {"type": "boolean"}}, ["pattern"],
              group="core", read_only=True, summary_key="pattern")
    def grep(ctx, pattern, path=".", glob="", ignore_case=False):
        try:
            rx = re.compile(pattern, re.I if truthy(ignore_case) else 0)
        except re.error as e:
            return "Error: invalid regex: %s" % e
        root = ctx.path(path)
        files = [root] if root.is_file() else walk_files(root)
        base = root.parent if root.is_file() else root
        grx = glob_to_regex(glob) if glob else None
        results = []
        for f in files:
            rel = f.relative_to(base).as_posix()
            if grx and not grx.match(rel):
                continue
            try:
                if f.stat().st_size > 2_000_000:
                    continue
                raw = f.read_bytes()
            except OSError:
                continue
            if b"\0" in raw[:8192]:
                continue
            for n, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
                if rx.search(line):
                    results.append("%s:%d:%s" % (rel, n, line.strip()[:300]))
                    if len(results) >= 200:
                        results.append("... (200 match limit)")
                        return "\n".join(results)
        return "\n".join(results) or "No matches."

    # ---- file management --------------------------------------------------
    @reg.tool("make_dir", "Create a directory (and parents).", {"path": {"type": "string"}}, ["path"],
              group="files", summary_key="path")
    def make_dir(ctx, path):
        ctx.path(path).mkdir(parents=True, exist_ok=True)
        return "Created directory %s" % ctx.path(path)

    @reg.tool("move_path", "Move or rename a file or directory.",
              {"src": {"type": "string"}, "dst": {"type": "string"}}, ["src", "dst"], group="files", ask=True, summary_key="src")
    def move_path(ctx, src, dst):
        s, d = ctx.path(src), ctx.path(dst)
        if not s.exists():
            return "Error: not found: %s" % s
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
        return "Moved %s -> %s" % (s, d)

    @reg.tool("copy_path", "Copy a file or directory.",
              {"src": {"type": "string"}, "dst": {"type": "string"}}, ["src", "dst"], group="files", ask=True, summary_key="src")
    def copy_path(ctx, src, dst):
        s, d = ctx.path(src), ctx.path(dst)
        if not s.exists():
            return "Error: not found: %s" % s
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            shutil.copytree(str(s), str(d), dirs_exist_ok=True)
        else:
            shutil.copy2(str(s), str(d))
        return "Copied %s -> %s" % (s, d)

    def _del_danger(a):
        return False

    @reg.tool("delete_path", "Delete a file or directory (recursive for directories). Cannot be undone.",
              {"path": {"type": "string"}}, ["path"], group="files", ask=True, summary_key="path")
    def delete_path(ctx, path):
        p = ctx.path(path)
        if p == Path(p.anchor) or p == Path.home().resolve():
            return "Error: refusing to delete a filesystem root or the home directory."
        if not p.exists():
            return "Error: not found: %s" % p
        shutil.rmtree(str(p)) if p.is_dir() and not p.is_symlink() else p.unlink()
        return "Deleted %s" % p

    @reg.tool("file_info", "Size, timestamps, line count, line endings and SHA-256 of a file.",
              {"path": {"type": "string"}}, ["path"], group="files", read_only=True, summary_key="path")
    def file_info(ctx, path):
        import datetime
        p = ctx.path(path)
        if not p.exists():
            return "Error: not found: %s" % p
        st = p.stat()
        info = ["path: %s" % p, "type: %s" % ("directory" if p.is_dir() else "file"), "size: %d bytes" % st.st_size,
                "modified: %s" % datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")]
        if p.is_file() and st.st_size < 200_000_000:
            raw = p.read_bytes()
            info.append("sha256: " + hashlib.sha256(raw).hexdigest())
            if b"\0" not in raw[:8192]:
                info.append("lines: %d" % (raw.count(b"\n") + (0 if raw.endswith(b"\n") or not raw else 1)))
                info.append("line endings: " + ("CRLF" if b"\r\n" in raw else "LF"))
        return "\n".join(info)

    @reg.tool("diff_files", "Unified diff between two text files.",
              {"a": {"type": "string"}, "b": {"type": "string"}}, ["a", "b"], group="files", read_only=True, summary_key="a")
    def diff_files(ctx, a, b):
        pa, pb = ctx.path(a), ctx.path(b)
        for p in (pa, pb):
            if not p.is_file():
                return "Error: not a file: %s" % p
        d = list(difflib.unified_diff(read_text_file(pa)[0].splitlines(), read_text_file(pb)[0].splitlines(),
                                      str(pa), str(pb), lineterm=""))
        return "\n".join(d) or "Files are identical."

    @reg.tool("archive", "Create, extract or list zip / tar.gz archives. action: create|extract|list.",
              {"action": {"type": "string", "enum": ["create", "extract", "list"]},
               "archive": {"type": "string", "description": "archive path (.zip, .tar.gz, .tgz, .tar)"},
               "source": {"type": "string", "description": "create: file/dir to add; extract: destination dir"}},
              ["action", "archive"], group="files", ask=lambda a: a.get("action") != "list", summary_key="archive")
    def archive(ctx, action, archive, source=""):
        ap = ctx.path(archive)
        name = ap.name.lower()
        is_zip = name.endswith(".zip")
        if action == "list":
            if is_zip:
                with zipfile.ZipFile(ap) as z:
                    return "\n".join("%10d  %s" % (i.file_size, i.filename) for i in z.infolist()[:500])
            with tarfile.open(ap) as t:
                return "\n".join("%10d  %s" % (m.size, m.name) for m in t.getmembers()[:500])
        if action == "create":
            src = ctx.path(source)
            if not src.exists():
                return "Error: source not found: %s" % src
            ap.parent.mkdir(parents=True, exist_ok=True)
            if is_zip:
                with zipfile.ZipFile(ap, "w", zipfile.ZIP_DEFLATED) as z:
                    if src.is_dir():
                        for f in walk_files(src):
                            z.write(f, f.relative_to(src.parent).as_posix())
                    else:
                        z.write(src, src.name)
            else:
                mode = "w:gz" if name.endswith((".gz", ".tgz")) else "w"
                with tarfile.open(ap, mode) as t:
                    t.add(src, arcname=src.name)
            return "Created %s" % ap
        if action == "extract":
            dest = ctx.path(source or ".")
            dest.mkdir(parents=True, exist_ok=True)
            root = dest.resolve()

            def safe(member_name):
                tgt = (root / member_name).resolve()
                return str(tgt).startswith(str(root))
            if is_zip:
                with zipfile.ZipFile(ap) as z:
                    bad = [n for n in z.namelist() if not safe(n)]
                    if bad:
                        return "Error: archive has unsafe paths (e.g. %s)." % bad[0]
                    z.extractall(root)
            else:
                with tarfile.open(ap) as t:
                    bad = [m.name for m in t.getmembers() if not safe(m.name) or m.issym() or m.islnk()]
                    if bad:
                        return "Error: archive has unsafe members (e.g. %s)." % bad[0]
                    t.extractall(root)
            return "Extracted to %s" % root
        return "Error: unknown action."


def plan_edit(ctx, name: str, args: dict):
    """For write_file / edit_file / multi_edit: return (path, old_text_or_None, new_text_or_None, error)."""
    p = ctx.path(args.get("path", ""))
    old = read_text_file(p)[0] if p.is_file() else None
    if name == "write_file":
        return p, old, str(args.get("content", "")).replace("\r\n", "\n"), None
    if name == "edit_file":
        edits = [(args.get("old_string", ""), args.get("new_string", ""), truthy(args.get("replace_all", False)))]
    else:
        raw = args.get("edits")
        if not isinstance(raw, list):
            return p, old, None, "edits must be a list"
        edits = [(e.get("old_string", ""), e.get("new_string", ""), truthy(e.get("replace_all"))) for e in raw if isinstance(e, dict)]
    if old is None:
        if name == "edit_file" and edits[0][0] == "":
            return p, None, edits[0][1], None
        return p, None, None, "file not found: %s" % p
    new, err = apply_edits(old, edits)
    return p, old, new, err
