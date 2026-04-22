from __future__ import annotations

import torch
import torch.nn.functional as F

from config import LossConfig, is_teacher


class LossWithMetrics:
    def __init__(self, name, loss_fn, metric_fn=None) -> None:
        self.name = name
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.last_metrics: dict[str, float] = {}

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(prediction, target)
        if self.metric_fn is None:
            raw_metrics = {"total_loss": loss, f"{self.name}_loss": loss}
        else:
            raw_metrics = self.metric_fn(prediction, target, loss)
            raw_metrics.setdefault("total_loss", loss)
        self.last_metrics = {
            key: float(value.detach().item()) if torch.is_tensor(value) else float(value)
            for key, value in raw_metrics.items()
        }
        return loss


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction, target)


def l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("TODO: implement l1_loss")


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    raise NotImplementedError("TODO: implement charbonnier_loss")


def mse_l1_loss(prediction: torch.Tensor, target: torch.Tensor, l1_weight: float = 0.2) -> torch.Tensor:
    raise NotImplementedError("TODO: implement mse_l1_loss")


def mse_edge_loss(prediction: torch.Tensor, target: torch.Tensor, edge_weight: float = 0.1) -> torch.Tensor:
    raise NotImplementedError("TODO: implement mse_edge_loss")


def build_loss(config: LossConfig):
    name = config.name

    if name == "mse":
        return LossWithMetrics(
            name="mse",
            loss_fn=mse_loss,
            metric_fn=lambda prediction, target, loss: {
                "total_loss": loss,
                "mse_loss": loss,
            },
        )

    if is_teacher():
        from _teacher_solutions import losses as ref

        if name == "l1":
            return LossWithMetrics(name="l1", loss_fn=ref.l1_loss)
        if name == "charbonnier":
            return LossWithMetrics(
                name="charbonnier",
                loss_fn=lambda p, t: ref.charbonnier_loss(p, t, eps=1e-3),
            )
        if name == "mse_l1":
            return LossWithMetrics(
                name="mse_l1",
                loss_fn=lambda p, t: ref.mse_l1_loss(p, t, l1_weight=0.2),
                metric_fn=lambda prediction, target, loss: {
                    "total_loss": loss,
                    "mse_loss": F.mse_loss(prediction, target),
                    "l1_loss": F.l1_loss(prediction, target),
                },
            )
        if name == "mse_edge":
            return LossWithMetrics(
                name="mse_edge",
                loss_fn=lambda p, t: ref.mse_edge_loss(p, t, edge_weight=0.1),
            )

    if name == "l1":
        return LossWithMetrics(name="l1", loss_fn=l1_loss)
    if name == "charbonnier":
        return LossWithMetrics(
            name="charbonnier",
            loss_fn=lambda prediction, target: charbonnier_loss(prediction, target, eps=1e-3),
        )
    if name == "mse_l1":
        return LossWithMetrics(
            name="mse_l1",
            loss_fn=lambda prediction, target: mse_l1_loss(prediction, target, l1_weight=0.2),
            metric_fn=lambda prediction, target, loss: {
                "total_loss": loss,
                "mse_loss": F.mse_loss(prediction, target),
                "l1_loss": F.l1_loss(prediction, target),
            },
        )
    if name == "mse_edge":
        return LossWithMetrics(
            name="mse_edge",
            loss_fn=lambda prediction, target: mse_edge_loss(prediction, target, edge_weight=0.1),
        )

    raise ValueError(f"Unknown loss name: {name}")
