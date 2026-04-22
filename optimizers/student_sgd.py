# ============================================================
# [STUDENT FILE] 你需要在此文件中实现 SGD 优化器
#
# 接口已定义好，你只需要实现 step() 方法中的参数更新逻辑。
# ============================================================
from __future__ import annotations

import torch


class StudentSGD:
    """
    A minimal hand-written SGD optimizer.

    SGD update rule:
        param = param - lr * grad
    """

    def __init__(self, params, lr: float) -> None:
        self.params = list(params)
        self.lr = lr

    def zero_grad(self) -> None:
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self) -> None:
        """
        TODO: implement the SGD parameter update.

        For each parameter with a gradient:
            param = param - lr * grad

        Hint: use torch.no_grad() context and in-place operations.
        """
        raise NotImplementedError(
            "TODO: implement SGD step(). See the docstring above for the update rule."
        )
