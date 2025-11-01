"""Utility functions for the pipeline."""

import torch
import numpy as np
from typing import Dict, Any


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        'trainable': trainable,
        'total': total,
        'frozen': total - trainable
    }


def log_metrics(metrics: Dict[str, Any], step: int = None):
    """Log metrics (placeholder for wandb integration)."""
    if step is not None:
        print(f"Step {step}:", end=" ")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}={value:.4f}", end=" ")
        else:
            print(f"{key}={value}", end=" ")
    print()

