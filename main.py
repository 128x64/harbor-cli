#!/usr/bin/env python3
"""Harbor CLI main script.

If a file named model.gguf sits in the same folder as this script, that local model is used (via llama-server / llama.cpp).
Otherwise NVIDIA NIM is used (set NVIDIA_API_KEY).   Run:  python main.py   |   python main.py --help
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from harbor.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(main_dir=HERE))
