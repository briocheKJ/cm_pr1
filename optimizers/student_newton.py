# ============================================================
# [STUDENT FILE] (bonus) 你可以在此文件中实现 Newton 风格优化器
#
# 使用对角 secant 近似估计 Hessian，实现阻尼 Newton 步。
# 这是一个进阶实现，不是必做项。
# ============================================================
from __future__ import annotations

import torch


class StudentNewton:
    """
    A teaching-friendly Newton-style optimizer (bonus).

    Uses diagonal secant approximation to curvature.

    TODO (bonus): implement this optimizer.
    """

    def __init__(
        self,
        params,
        lr: float,
        damping: float = 1e-3,
        max_curvature: float = 1e3,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.damping = damping
        self.max_curvature = max_curvature

    def zero_grad(self) -> None:
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self) -> None:
        raise NotImplementedError("TODO (bonus): implement Newton step().")
