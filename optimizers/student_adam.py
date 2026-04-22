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
    """

    def __init__(
        self,
        params,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_count = 0

        # First moment (mean of gradients) and second moment (mean of squared gradients).
        self.m = [torch.zeros_like(param) for param in self.params]
        self.v = [torch.zeros_like(param) for param in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self) -> None:
        """
        TODO: implement the Adam parameter update.

        Steps:
        1. Increment self.step_count
        2. For each parameter with a gradient:
           a. Update first moment:  m = beta1 * m + (1 - beta1) * grad
           b. Update second moment: v = beta2 * v + (1 - beta2) * grad^2
           c. Bias correction:      m_hat = m / (1 - beta1^t)
                                    v_hat = v / (1 - beta2^t)
           d. Update parameter:     param -= lr * m_hat / (sqrt(v_hat) + eps)

        Hint: use torch.no_grad() context and in-place operations.
        """
        raise NotImplementedError(
            "TODO: implement Adam step(). See the docstring above for the update rule."
        )
