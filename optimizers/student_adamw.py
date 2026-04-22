# ============================================================
# [STUDENT FILE] 你需要在此文件中实现 AdamW 优化器
#
# AdamW 与 Adam 的区别：权重衰减是解耦的（直接作用在参数上）。
# ============================================================
from __future__ import annotations

import torch


class StudentAdamW:
    """
    A minimal hand-written AdamW optimizer.

    AdamW update rule (per parameter):
        param = param * (1 - lr * weight_decay)      # decoupled weight decay
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        param = param - lr * m_hat / (sqrt(v_hat) + eps)

    Note: each parameter group has its own learning rate (group["lr"]).
    """

    def __init__(
        self,
        param_groups: list[dict],
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        self.param_groups = param_groups
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0

        self.state: dict[int, dict[str, torch.Tensor]] = {}
        for group in self.param_groups:
            for param in group["params"]:
                self.state[id(param)] = {
                    "m": torch.zeros_like(param),
                    "v": torch.zeros_like(param),
                }

    def zero_grad(self) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.zero_()

    def step(self) -> None:
        """
        TODO: implement the AdamW parameter update.

        Key difference from Adam: apply weight decay BEFORE the adaptive update.

        Hint: see the docstring above for the update rule.
        """
        raise NotImplementedError(
            "TODO: implement AdamW step(). See the docstring above for the update rule."
        )
