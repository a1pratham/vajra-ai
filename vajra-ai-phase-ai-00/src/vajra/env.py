"""Environment reporting for VAJRA-AI.

Prints/returns Python version, PyTorch version, CUDA availability, GPU name,
and GPU memory. GPU is detected but never required — every later phase must
degrade gracefully to CPU, and this module is where that convention starts.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional

import torch


def get_env_report() -> Dict[str, Any]:
    """Return a dict describing the current environment."""
    cuda_available = torch.cuda.is_available()

    report: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "gpu_name": "N/A — running on CPU",
        "gpu_memory_gb": "N/A — running on CPU",
    }

    if cuda_available:
        try:
            report["gpu_name"] = torch.cuda.get_device_name(0)
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            report["gpu_memory_gb"] = round(total_bytes / (1024 ** 3), 1)
        except Exception as exc:  # pragma: no cover - defensive only
            report["gpu_name"] = f"Detected but could not query details ({exc})"
            report["gpu_memory_gb"] = "N/A"

    return report


def print_env_report(report: Optional[Dict[str, Any]] = None) -> None:
    """Pretty-print the environment report."""
    if report is None:
        report = get_env_report()

    print("=== VAJRA-AI Environment Report ===")
    print(f"Python version   : {report['python_version']}")
    print(f"PyTorch version  : {report['torch_version']}")
    print(f"CUDA available   : {report['cuda_available']}")
    print(f"GPU name         : {report['gpu_name']}")
    print(f"GPU memory (GB)  : {report['gpu_memory_gb']}")
    print("====================================")


def get_device() -> torch.device:
    """Return cuda device if available, else cpu. No hard GPU requirement."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    print_env_report()
