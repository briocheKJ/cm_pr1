from __future__ import annotations

from collections.abc import Iterable

import torch

from config import TorchAdamConfig


def build_torch_adam(
    params: Iterable[torch.nn.Parameter],
    config: TorchAdamConfig,
) -> torch.optim.Optimizer:
    """
    Baseline optimizer used by the starter code.
    """
    return torch.optim.Adam(
        params,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )
