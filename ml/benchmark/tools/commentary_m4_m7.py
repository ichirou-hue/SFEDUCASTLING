#!/usr/bin/env python3
"""CLI for the automatic M4-M7 commentary benchmark."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gigabench.commentary import main  # noqa: E402


if __name__ == "__main__":
    main()
