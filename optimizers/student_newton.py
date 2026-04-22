# ============================================================
# [STUDENT FILE] (bonus) 你可以在此文件中实现 Newton 风格优化器
# ============================================================
from __future__ import annotations

import torch


class StudentNewton:
    """
    A teaching-friendly Newton-style optimizer (bonus).

    TODO (bonus): implement this optimizer.
    """

    def __init__(
        self,
        param_groups: list[dict],
        damping: float = 1e-3,
        max_curvature: float = 1e3,
    ) -> None:
        self.param_groups = param_groups
        self.damping = damping
        self.max_curvature = max_curvature

    def zero_grad(self) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.zero_()

    def step(self) -> None:
        raise NotImplementedError("TODO (bonus): implement Newton step().")
