"""Load the model into memory BEFORE the first message (so message #1 is as fast as message #2).

* Ollama (local): an empty /api/generate request loads it, with the same num_ctx the chat requests will use.
* llama-server / LM Studio / vLLM on this machine: one 1-token request that carries the real system prompt and tool list, which
  also fills the server's prompt cache.
* llama-cpp-python: one generated token (the weights are already in this process; this pages them in).
* Remote services (NVIDIA NIM, OpenAI, ...) have nothing to load, so nothing is done.
"""
from __future__ import annotations

import time
import urllib.parse

from .backends import LlamaCppPython
from .client import NIMClient
from .models import unwrap
from .ollama import OllamaClient


def is_loopback(url: str) -> bool:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0") or host.startswith("127.")


def preload(agent):
    """Returns seconds spent, or None when the backend has nothing to preload. May raise NIMError."""
    c = unwrap(agent.client)
    t0 = time.time()
    if isinstance(c, OllamaClient) and is_loopback(c.host):
        c.preload(agent.model)
    elif isinstance(c, LlamaCppPython):
        c.warm()
    elif isinstance(c, NIMClient) and is_loopback(c.base_url):
        specs = [t.spec() for t in agent.active_tools()]
        agent.client.chat_stream(agent.model, [agent.messages[0], {"role": "user", "content": "Reply with one word: ready"}],
                                 specs, 1, 0.0)
    else:
        return None
    return time.time() - t0
