"""Global seed management for reproducibility.

Sets seeds for Python stdlib random, numpy, and torch (if available).
Logs the seed to MLflow if a run is active.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value. Logged to MLflow if a run is active.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Torch seeding (optional — only if torch is installed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except ImportError:
        pass

    # Log to MLflow if a run is active
    try:
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.log_param("global_seed", seed)
    except ImportError:
        pass
