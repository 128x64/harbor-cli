"""Built-in tool groups. Each module exposes register(reg)."""
from . import agentic, browser, code, fs, media, shell, system, web


def register_all(reg) -> None:
    for mod in (fs, shell, web, code, media, system, agentic, browser):
        mod.register(reg)
