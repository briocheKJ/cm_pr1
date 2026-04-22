# ============================================================
# [STUDENT FILE] (bonus) 你可以在此文件中实现 Muon 优化器
# ============================================================
from __future__ import annotations

import torch


class StudentMuon:
    """
    A teaching-friendly Muon-style optimizer (bonus).

    TODO (bonus): implement this optimizer.
    """

    def __init__(
        self,
        param_groups: list[dict],
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        nesterov: bool = True,
    ) -> None:
        self.param_groups = param_groups
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.ns_steps = ns_steps
        self.nesterov = nesterov
        self.buffers: dict[int, torch.Tensor] = {}
        for group in self.param_groups:
            for param in group["params"]:
                self.buffers[id(param)] = torch.zeros_like(param)

    def zero_grad(self) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.zero_()

    def step(self) -> None:
        raise NotImplementedError("TODO (bonus): implement Muon step().")
