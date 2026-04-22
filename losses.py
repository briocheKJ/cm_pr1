# ============================================================
# [STUDENT FILE] 你需要在此文件中实现新的 loss 函数
#
# 已提供: mse_loss (基线)
# 需实现: l1_loss, charbonnier_loss, mse_l1_loss, mse_edge_loss
# ============================================================
from __future__ import annotations

import torch
import torch.nn.functional as F

from config import LossConfig


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Default RGB reconstruction loss.
    """
    return F.mse_loss(prediction, target)


def l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    L1 (absolute error) loss.

    TODO: implement this loss function.
    Hint: you can use F.l1_loss or compute it manually.
    """
    raise NotImplementedError("TODO: implement l1_loss")


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    Charbonnier loss — a smooth approximation to L1.

    Formula: mean( sqrt( (pred - target)^2 + eps^2 ) )

    TODO: implement this loss function.
    """
    raise NotImplementedError("TODO: implement charbonnier_loss")


def mse_l1_loss(prediction: torch.Tensor, target: torch.Tensor, l1_weight: float = 0.2) -> torch.Tensor:
    """
    Combined MSE + L1 loss.

    Formula: MSE(pred, target) + l1_weight * L1(pred, target)

    TODO: implement this loss function.
    """
    raise NotImplementedError("TODO: implement mse_l1_loss")


def mse_edge_loss(prediction: torch.Tensor, target: torch.Tensor, edge_weight: float = 0.1) -> torch.Tensor:
    """
    MSE + Sobel edge matching loss.

    Formula: MSE(pred, target) + edge_weight * L1(sobel(pred), sobel(target))

    TODO: implement this loss function.
    Hint: you need to implement a Sobel edge detector first.
    """
    raise NotImplementedError("TODO: implement mse_edge_loss")


def build_loss(config: LossConfig):
    """
    Build a training loss from a short name.

    Supported:
    - mse        (provided)
    - l1         (student TODO)
    - charbonnier(student TODO)
    - mse_l1     (student TODO)
    - mse_edge   (student TODO)
    """
    from mode import is_teacher

    name = config.name

    if name == "mse":
        return mse_loss

    # In teacher mode, load reference implementations.
    if is_teacher():
        from _teacher_solutions import losses as ref

        if name == "l1":
            return ref.l1_loss
        if name == "charbonnier":
            return lambda p, t: ref.charbonnier_loss(p, t, eps=config.charbonnier_eps)
        if name == "mse_l1":
            return lambda p, t: ref.mse_l1_loss(p, t, l1_weight=config.l1_weight)
        if name == "mse_edge":
            return lambda p, t: ref.mse_edge_loss(p, t, edge_weight=config.edge_weight)

    # Student mode — use the functions defined above (stubs until implemented).
    if name == "l1":
        return l1_loss

    if name == "charbonnier":
        return lambda prediction, target: charbonnier_loss(
            prediction, target, eps=config.charbonnier_eps,
        )

    if name == "mse_l1":
        return lambda prediction, target: mse_l1_loss(
            prediction, target, l1_weight=config.l1_weight,
        )

    if name == "mse_edge":
        return lambda prediction, target: mse_edge_loss(
            prediction, target, edge_weight=config.edge_weight,
        )

    raise ValueError(f"Unknown loss name: {name}")
