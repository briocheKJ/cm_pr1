# ============================================================
# [STUDENT FILE] 你需要在此文件中实现 Adam 优化器
#
# 接口已定义好，你只需要实现 step() 方法中的参数更新逻辑。
# ============================================================
from __future__ import annotations

import torch


class StudentAdam:
    """
    A minimal hand-written Adam optimizer.

    Adam update rule (per parameter):
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
    ) -> None:
        self.param_groups = param_groups
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_count = 0

        # Per-parameter state: first moment (m) and second moment (v).
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
        TODO: implement the Adam parameter update.

        Steps:
        1. Increment self.step_count
        2. For each parameter group (use group["lr"] as the learning rate):
           For each parameter with a gradient:
               a. Get state: s = self.state[id(param)]
               b. Update first moment:  s["m"] = beta1 * s["m"] + (1 - beta1) * grad
               c. Update second moment: s["v"] = beta2 * s["v"] + (1 - beta2) * grad^2
               d. Bias correction:      m_hat = s["m"] / (1 - beta1^t)
                                         v_hat = s["v"] / (1 - beta2^t)
               e. Update parameter:     param -= lr * m_hat / (sqrt(v_hat) + eps)

        Hint: use torch.no_grad() context.
        """
        raise NotImplementedError(
            "TODO: implement Adam step(). See the docstring above for the update rule."
        )
