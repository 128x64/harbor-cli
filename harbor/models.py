"""Model profiles: small adaptations that make very different models behave well inside the same agent harness
(sampling quirks, API parameter names, extra guidance for small models)."""
from __future__ import annotations

import re
from dataclasses import dataclass

SMALL_GUIDE = (
    "\n\n# Extra guidance (small model)\n"
    "- Work in small steps: after every tool result decide ONE next action.\n"
    "- Never describe a change instead of making it - use the tools.\n"
    "- Emit tool calls exactly in the required format and double-check argument names against the tool list.\n"
    "- Read files before editing; make the smallest edit that works; run the tests after.\n"
    "- Keep final answers short.")


@dataclass
class Profile:
    name: str = "default"
    temperature: float | None = None      # default sampling temperature (None = harness default)
    omit_temperature: bool = False        # some reasoning models reject a temperature parameter
    max_tokens_key: str = "max_tokens"    # OpenAI reasoning models want max_completion_tokens
    guidance: str = ""                    # appended to the system prompt
    matched: tuple = ()


# (regex on the lower-cased model id, field overrides)
RULES = [
    ("openai-reasoning", r"(^|[/:_-])(o\d(-|$)|gpt-5)", dict(omit_temperature=True, max_tokens_key="max_completion_tokens")),
    ("deepseek-reasoner", r"deepseek.*(r1|reasoner)", dict(temperature=0.6)),
    ("qwen3", r"qwen3", dict(temperature=0.6)),
    ("small-model", r"(^|[/:_-])(0\.\d|1|2|3|4|7|8|9)b($|[-_:.])", dict(guidance=SMALL_GUIDE)),
]


def profile_for(model: str) -> Profile:
    p, names = Profile(), []
    for name, rx, fields in RULES:
        if re.search(rx, (model or "").lower()):
            for k, v in fields.items():
                setattr(p, k, v)
            names.append(name)
    p.matched = tuple(names)
    if names:
        p.name = "+".join(names)
    return p


def unwrap(client):
    """Return the innermost HTTP client behind TextToolClient / AutoToolClient wrappers."""
    seen = 0
    while seen < 5 and (hasattr(client, "inner") or hasattr(client, "native")):
        client = getattr(client, "inner", None) or client.native
        seen += 1
    return client


def apply_profile(client, profile: Profile) -> None:
    c = unwrap(client)
    c.omit_temperature = profile.omit_temperature
    c.max_tokens_key = profile.max_tokens_key
