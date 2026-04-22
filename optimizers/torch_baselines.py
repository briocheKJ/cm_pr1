from __future__ import annotations

import torch

from config import TorchAdamConfig


def build_torch_adam(
    param_groups: list[dict],
    config: TorchAdamConfig,
) -> torch.optim.Optimizer:
    """
    Baseline optimizer used by the starter code.

    Accepts param_groups with per-group learning rates.
    """
    # Convert our param_groups format to PyTorch's format.
    torch_groups = []
    for group in param_groups:
        torch_groups.append({
            "params": group["params"],
            "lr": group["lr"],
            "base_lr": group["base_lr"],
        })
    return torch.optim.Adam(
        torch_groups,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )
