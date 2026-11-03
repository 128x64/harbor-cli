"""Code execution and data tools: Python, SQLite, safe calculator."""
from __future__ import annotations

import ast
import math
import operator
import os
import sqlite3
import sys
import tempfile

from ..core import Registry
from ..util import bold, cyan, truncate
from .shell import run_command
from ..gitfinder import tool_env

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.UAdd: operator.pos}
_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
_NAMES.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})


def safe_eval(expr: str):
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            l, r = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(r) > 10000:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](l, r)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Name) and n.id in _NAMES and not callable(_NAMES[n.id]):
            return _NAMES[n.id]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and callable(_NAMES.get(n.func.id)) and not n.keywords:
            return _NAMES[n.func.id](*[ev(a) for a in n.args])
        if isinstance(n, (ast.Tuple, ast.List)):
            return [ev(e) for e in n.elts]
        raise ValueError("unsupported expression")
    return ev(ast.parse(expr.strip(), mode="eval"))


def _sql_mutating(a: dict) -> bool:
    first = (a.get("sql", "").lstrip().split(None, 1) or [""])[0].lower()
    return first not in ("select", "pragma", "explain", "with")


def register(reg: Registry) -> None:
    @reg.tool("python_exec", "Run a Python snippet with the same interpreter that runs nimcode (all pip packages "
                             "available). Use print() for output. Good for data processing, quick experiments, generating files.",
              {"code": {"type": "string"}, "timeout": {"type": "integer", "description": "seconds (default 60, max 600)"}},
              ["code"], group="code", ask=True, summary_key="code",
              preview=lambda ctx, a: "\n".join(str(a.get("code", "")).splitlines()[:25]))
    def python_exec(ctx, code, timeout=60):
        fd, path = tempfile.mkstemp(suffix=".py", prefix="nimcode_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(code)
            env = dict(tool_env(), PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
            return run_command([sys.executable, "-u", path], ctx.cwd, min(int(timeout or 60), 600), env)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    @reg.tool("sqlite_query", "Run SQL against a SQLite database file. SELECT/PRAGMA/EXPLAIN are read-only and need no approval.",
              {"db_path": {"type": "string"}, "sql": {"type": "string"}}, ["db_path", "sql"], group="code",
              ask=_sql_mutating, summary_key="sql")
    def sqlite_query(ctx, db_path, sql):
        p = ctx.path(db_path)
        mutating = _sql_mutating({"sql": sql})
        if not p.exists() and not mutating:
            return "Error: database not found: %s" % p
        try:
            if mutating:
                con = sqlite3.connect(str(p))
            else:
                con = sqlite3.connect("file:%s?mode=ro" % p.as_posix().replace("?", "%3f"), uri=True)
            cur = con.cursor()
            cur.execute(sql)
            if cur.description:
                cols = [d[0] for d in cur.description]
                rows = cur.fetchmany(200)
                out = "\t".join(cols) + "\n" + "\n".join("\t".join("" if v is None else str(v) for v in r) for r in rows)
                more = cur.fetchone()
                return truncate(out + ("\n... (more rows)" if more else ""))
            con.commit()
            return "OK (%d rows affected)" % cur.rowcount
        except sqlite3.Error as e:
            return "SQL error: %s" % e
        finally:
            try:
                con.close()
            except Exception:
                pass

    @reg.tool("calculate", "Evaluate a math expression safely (+ - * / // % **, math functions like sqrt, sin, log, constants pi, e).",
              {"expression": {"type": "string"}}, ["expression"], group="code", read_only=True, summary_key="expression")
    def calculate(ctx, expression):
        try:
            return str(safe_eval(expression))
        except Exception as e:
            return "Error: %s" % e
