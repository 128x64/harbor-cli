"""Custom tool example: copy to ~/.nimcode/tools/ (global) or <project>/.nimcode/tools/ (asks for trust)."""
from pathlib import Path


def word_count(ctx, path):
    text = ctx.path(path).read_text(encoding="utf-8", errors="replace")
    return "%d words, %d lines" % (len(text.split()), len(text.splitlines()))


TOOLS = [{
    "name": "word_count",
    "description": "Count words and lines in a text file.",
    "parameters": {"path": {"type": "string"}},
    "required": ["path"],
    "func": word_count,
    "ask": False,
    "read_only": True,
}]
