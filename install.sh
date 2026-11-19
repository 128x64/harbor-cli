#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install -r requirements-local.txt || echo "(llama-cpp-python skipped - use llama-server instead)"
python -m pip install -r requirements-extra.txt || echo "(optional extras skipped)"
echo "Done. export NVIDIA_API_KEY=nvapi-...  or put model.gguf next to main.py, then: .venv/bin/python main.py"
