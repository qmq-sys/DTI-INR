"""Run Spatial DTI-INR (Model A) experiment."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_a import main

if __name__ == "__main__":
    main()
