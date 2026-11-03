#!/usr/bin/env python3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nimcode.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(main_dir=HERE))
