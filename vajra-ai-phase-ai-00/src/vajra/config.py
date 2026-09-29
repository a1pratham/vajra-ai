"""Config loading for VAJRA-AI.

Keeps this deliberately simple: read configs/config.yaml into a plain dict.
Later phases may wrap this in a dataclass if the config grows enough to
warrant it — not needed yet.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = os.path.join("configs", "config.yaml")


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load the YAML config file and return it as a dict.

    Raises FileNotFoundError with a clear message if the file is missing,
    since a missing config is a common first failure mode when the project
    folder structure got flattened during upload/unzip.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find config at '{path}'. Make sure you unzipped the "
            "whole vajra-ai-phase-ai-00 folder (not just one subfolder) and "
            "that your working directory is the project root."
        )
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config at '{path}' did not parse to a dict.")
    return config
