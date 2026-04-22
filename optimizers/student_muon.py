# ============================================================
# [STUDENT FILE] (bonus) 你可以在此文件中实现 Muon 优化器
#
# Muon 是一种使用 Newton-Schulz 正交化的优化器。
# 这是一个进阶实现，不是必做项。
# ============================================================
from __future__ import annotations

import torch


class StudentMuon:
    """
    A teaching-friendly Muon-style optimizer (bonus).

    Muon uses momentum + Newton-Schulz orthogonalization for matrix parameters.

    TODO (bonus): implement this optimizer.
    """

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        nesterov: bool = True,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.ns_steps = ns_steps
        self.nesterov = nesterov
        self.buffers = [torch.zeros_like(param) for param in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self) -> None:
        raise NotImplementedError("TODO (bonus): implement Muon step().")
