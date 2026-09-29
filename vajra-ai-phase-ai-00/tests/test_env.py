"""Basic tests for Phase AI-0.

Run with: pytest tests/test_env.py -v
(from the project root, with src/ on the path — see conftest-free approach
below using sys.path, kept simple on purpose for a hackathon project).
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "configs", "config.yaml")

from vajra.config import load_config
from vajra.env import get_env_report, get_device
from vajra.seed import check_reproducibility
import torch


def test_config_loads():
    config = load_config(_CONFIG_PATH)
    assert "seed" in config
    assert "project" in config
    assert config["project"]["name"] == "vajra-ai"


def test_env_report_has_required_keys():
    report = get_env_report()
    for key in ("python_version", "torch_version", "cuda_available", "gpu_name", "gpu_memory_gb"):
        assert key in report


def test_reproducibility():
    assert check_reproducibility(42) is True


def test_device_forward_pass():
    device = get_device()
    layer = torch.nn.Linear(8, 2).to(device)
    x = torch.randn(3, 8).to(device)
    y = layer(x)
    assert y.shape == (3, 2)
