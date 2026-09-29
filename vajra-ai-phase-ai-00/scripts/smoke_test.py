"""VAJRA-AI — Phase AI-0 smoke test.

Run before anything else: `python scripts/smoke_test.py`

Verifies, in order:
  1. Core imports work (torch, numpy, yaml)
  2. Config loads from configs/config.yaml
  3. Seeding is reproducible
  4. A minimal tensor forward pass runs on the detected device

Exits with a non-zero status code on failure so this can be used in CI later.
"""

from __future__ import annotations

import os
import sys

# Make sure `src/` is importable whether this is run from the project root
# or from within scripts/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Config path is relative to project root, not to this script.
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "configs", "config.yaml")


def main() -> int:
    # Step 0: environment report (informational, not a pass/fail step)
    try:
        from vajra.env import print_env_report, get_device
    except Exception as exc:
        print(f"[FAIL] Could not import vajra.env: {exc}")
        return 1

    print_env_report()
    print()

    # Step 1: imports
    try:
        import torch  # noqa: F401
        import numpy  # noqa: F401
        import yaml  # noqa: F401
        print("[1/4] Imports OK")
    except Exception as exc:
        print(f"[1/4] Imports FAILED: {exc}")
        return 1

    # Step 2: config
    try:
        from vajra.config import load_config
        config = load_config(_CONFIG_PATH)
        assert "seed" in config, "config missing 'seed' key"
        assert "project" in config, "config missing 'project' key"
        print("[2/4] Config loaded OK")
    except Exception as exc:
        print(f"[2/4] Config loading FAILED: {exc}")
        return 1

    # Step 3: reproducibility
    try:
        from vajra.seed import check_reproducibility
        seed = config.get("seed", 42)
        reproducible = check_reproducibility(seed)
        assert reproducible, "seeded runs did not match"
        print(f"[3/4] Seed set OK (reproducibility check passed, seed={seed})")
    except Exception as exc:
        print(f"[3/4] Reproducibility check FAILED: {exc}")
        return 1

    # Step 4: minimal forward pass on detected device
    try:
        import torch

        device = get_device()
        layer = torch.nn.Linear(16, 4).to(device)
        dummy_input = torch.randn(2, 16).to(device)
        output = layer(dummy_input)
        assert output.shape == (2, 4), f"unexpected output shape {output.shape}"
        print(f"[4/4] Tensor forward pass OK on device: {device}")
    except Exception as exc:
        print(f"[4/4] Tensor forward pass FAILED: {exc}")
        return 1

    print()
    print("SMOKE TEST: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
