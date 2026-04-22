# ============================================================
# [STUDENT FILE] 你需要在此文件中实现 SGD + Momentum 优化器
#
# Momentum 是 SGD 和 Adam 之间的重要过渡：
#   SGD:      param -= lr * grad
#   Momentum: v = momentum * v + grad
#             param -= lr * v
#   Adam:     uses adaptive per-parameter learning rates
# ============================================================
from __future__ import annotations

import torch


class StudentMomentum:
    """
    SGD with momentum.

    Momentum update rule (per parameter):
        v = momentum * v + grad
        param = param - lr * v

    This is the "classical" momentum formulation.
    """

    def __init__(self, param_groups: list[dict], momentum: float = 0.9) -> None:
        self.param_groups = param_groups
        self.momentum = momentum

        # Velocity buffer for each parameter.
        self.velocity: dict[int, torch.Tensor] = {}
        for group in self.param_groups:
            for param in group["params"]:
                self.velocity[id(param)] = torch.zeros_like(param)

    def zero_grad(self) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.zero_()

    def step(self) -> None:
        """
        TODO: implement SGD + Momentum update.

        For each parameter group (use group["lr"] as the learning rate):
            For each parameter in the group:
                1. v = momentum * v + grad
                2. param -= lr * v

        Use self.velocity[id(param)] to access the velocity buffer.
        Hint: use torch.no_grad() context.
        """
        raise NotImplementedError(
            "TODO: implement Momentum step(). See the docstring above for the update rule."
        )
