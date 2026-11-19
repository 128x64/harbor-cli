"""Provider presets: one OpenAI-compatible client (client.NIMClient) can talk to almost any hosted or local model server."""
from __future__ import annotations

import os

# name -> (label, base_url, api-key env vars, default model, needs_key)
PROVIDERS = {
    "nim": ("NVIDIA NIM", "https://integrate.api.nvidia.com/v1", ("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY", "NIM_API_KEY"),
            "nvidia/nemotron-3-super-120b-a12b", True),
    "openai": ("OpenAI", "https://api.openai.com/v1", ("OPENAI_API_KEY",), "gpt-4.1", True),
    "anthropic": ("Anthropic (OpenAI-compat layer)", "https://api.anthropic.com/v1", ("ANTHROPIC_API_KEY",), "claude-sonnet-5-5", True),
    "gemini": ("Google Gemini (OpenAI-compat)", "https://generativelanguage.googleapis.com/v1beta/openai", ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
               "gemini-2.5-flash", True),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1", ("OPENROUTER_API_KEY",), "qwen/qwen3-coder", True),
    "groq": ("Groq", "https://api.groq.com/openai/v1", ("GROQ_API_KEY",), "llama-3.3-70b-versatile", True),
    "together": ("Together AI", "https://api.together.xyz/v1", ("TOGETHER_API_KEY",), "Qwen/Qwen2.5-Coder-32B-Instruct", True),
    "deepseek": ("DeepSeek", "https://api.deepseek.com/v1", ("DEEPSEEK_API_KEY",), "deepseek-chat", True),
    "mistral": ("Mistral", "https://api.mistral.ai/v1", ("MISTRAL_API_KEY",), "mistral-large-latest", True),
    "xai": ("xAI", "https://api.x.ai/v1", ("XAI_API_KEY",), "grok-4", True),
    "fireworks": ("Fireworks", "https://api.fireworks.ai/inference/v1", ("FIREWORKS_API_KEY",), "accounts/fireworks/models/qwen3-coder-480b-a35b-instruct", True),
    "lmstudio": ("LM Studio (local)", "http://localhost:1234/v1", (), "", False),
    "vllm": ("vLLM (local server)", "http://localhost:8000/v1", ("VLLM_API_KEY",), "", False),
    "llamacpp": ("llama.cpp server", "http://localhost:8080/v1", (), "", False),
    "custom": ("Custom OpenAI-compatible", "", ("HARBOR_API_KEY", "NIMCODE_API_KEY", "OPENAI_API_KEY"), "", False),
}


def provider_info(name: str):
    if name not in PROVIDERS:
        raise KeyError("unknown provider '%s' (choose from: %s)" % (name, ", ".join(PROVIDERS)))
    label, url, envs, model, needs_key = PROVIDERS[name]
    return {"label": label, "base_url": url, "key_env": envs, "model": model, "needs_key": needs_key}


def find_key(name: str, cfg: dict, key_env: str = "", key: str = ""):
    """Key lookup order: --api-key, --api-key-env, the provider's usual env vars, config 'keys'/'api_key'."""
    if key:
        return key
    envs = ([key_env] if key_env else []) + list(provider_info(name)["key_env"])
    for e in envs:
        if os.environ.get(e):
            return os.environ[e].strip()
    k = (cfg.get("keys") or {}).get(name)
    if k:
        return str(k).strip()
    if name == "nim" and cfg.get("api_key"):
        return str(cfg["api_key"]).strip()
    return None
