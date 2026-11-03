"""Agent-level tools: todo list, ask the user, persistent memory, sub-agents."""
from __future__ import annotations

from pathlib import Path

from ..core import Registry
from ..util import bold, cyan

MEMORY_FILE = Path.home() / ".nimcode" / "memory.md"


def register(reg: Registry) -> None:
    @reg.tool("todo_write", "Create/update the task checklist for multi-step work. Always send the full list.",
              {"todos": {"type": "array", "items": {"type": "object", "properties": {
                  "content": {"type": "string"},
                  "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}},
                  "required": ["content", "status"]}}}, ["todos"], group="core", read_only=True)
    def todo_write(ctx, todos):
        if not isinstance(todos, list):
            return "Error: todos must be a list of {content, status}."
        clean = []
        for t in todos:
            if isinstance(t, dict) and t.get("content"):
                st = t.get("status", "pending")
                clean.append({"content": str(t["content"]), "status": st if st in ("pending", "in_progress", "completed") else "pending"})
        ctx.todos = clean
        return "Todo list updated (%d items)." % len(clean)

    @reg.tool("ask_user", "Ask the human a clarifying question and wait for the answer. Use when requirements are ambiguous.",
              {"question": {"type": "string"}}, ["question"], group="core", read_only=True, summary_key="question")
    def ask_user(ctx, question):
        if not ctx.interactive:
            return "No interactive user is available. Proceed with your best judgment and state your assumptions."
        from ..ui import ui
        with ui.paused():
            try:
                return input(bold("? ") + question + "\n> ").strip() or "(no answer)"
            except (EOFError, KeyboardInterrupt):
                print()
                return "(the user did not answer)"

    @reg.tool("remember", "Save a durable fact or preference to long-term memory (~/.nimcode/memory.md, loaded into future sessions). "
                          "Only for things the user wants remembered; never secrets.",
              {"note": {"type": "string"}}, ["note"], group="core", ask=True, summary_key="note")
    def remember(ctx, note):
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write("- %s\n" % note.strip().replace("\n", " "))
        return "Saved to %s" % MEMORY_FILE

    @reg.tool("task",
              "Delegate a self-contained research/exploration job to a sub-agent with a fresh context (read-only tools "
              "+ web). It returns a concise report. Use for broad codebase searches or investigations so your own context stays small.",
              {"description": {"type": "string", "description": "3-6 word title"},
               "prompt": {"type": "string", "description": "complete instructions; the sub-agent sees nothing else"}},
              ["description", "prompt"], group="core", read_only=True, summary_key="description")
    def task(ctx, description, prompt):
        if ctx.agent is None:
            return "Error: no agent available."
        return ctx.agent.run_subagent(description, prompt)


    @reg.tool("finish",
              "AUTONOMOUS MODE ONLY: declare the goal complete (or impossible). Call it only after you verified the result "
              "(tests/build/run). summary = what was done and how it was verified; success=false if blocked.",
              {"summary": {"type": "string"}, "success": {"type": "boolean"}}, ["summary"], group="core", read_only=False,
              summary_key="summary")
    def finish(ctx, summary, success=True):
        if ctx.agent is not None:
            ctx.agent._finished = True
            ctx.agent._finish_ok = str(success).lower() not in ("false", "0", "no")
            ctx.agent._finish_summary = summary
        return summary

    @reg.tool("delegate",
              "Hand a self-contained sub-task to a worker sub-agent with a fresh context and ALL tools (it can edit files and "
              "run commands). Use it to keep your own context small on long jobs. Give complete instructions; it sees nothing "
              "else. Returns its report. Runs one at a time.",
              {"description": {"type": "string", "description": "3-6 word title"},
               "prompt": {"type": "string", "description": "complete instructions incl. paths and acceptance criteria"}},
              ["description", "prompt"], group="core", summary_key="description")
    def delegate(ctx, description, prompt):
        if ctx.agent is None:
            return "Error: no agent available."
        return ctx.agent.run_subagent(description, prompt, worker=True)
