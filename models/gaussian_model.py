from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(x / (1.0 - x))


def inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(y))


@dataclass
class GaussianRenderParams:
    centers: torch.Tensor
    scales: torch.Tensor
    angles: torch.Tensor
    alphas: torch.Tensor
    colors: torch.Tensor


class Gaussian2DModel(nn.Module):
    """
    Minimal Gaussian parameter container.

    This class is intentionally small:
    - it stores raw trainable parameters
    - it applies simple constraints
    - it returns the current render-ready values

    Initialization is handled elsewhere so students can swap strategies easily.
    """

    def __init__(self, num_gaussians: int = 32) -> None:
        super().__init__()
        self.num_gaussians = num_gaussians

        self.center_raw = nn.Parameter(torch.zeros(num_gaussians, 2))
        self.scale_raw = nn.Parameter(torch.zeros(num_gaussians, 2))
        self.angle_raw = nn.Parameter(torch.zeros(num_gaussians, 1))
        self.alpha_raw = nn.Parameter(torch.zeros(num_gaussians, 1))
        self.color_raw = nn.Parameter(torch.zeros(num_gaussians, 3))

    def get_render_params(self) -> GaussianRenderParams:
        centers = torch.sigmoid(self.center_raw)
        scales = F.softplus(self.scale_raw) + 1e-4
        angles = torch.pi * torch.tanh(self.angle_raw)
        alphas = torch.sigmoid(self.alpha_raw)
        colors = torch.sigmoid(self.color_raw)
        return GaussianRenderParams(
            centers=centers,
            scales=scales,
            angles=angles,
            alphas=alphas,
            colors=colors,
        )

    def set_raw_parameters(
        self,
        center_raw: torch.Tensor,
        scale_raw: torch.Tensor,
        angle_raw: torch.Tensor,
        alpha_raw: torch.Tensor,
        color_raw: torch.Tensor,
    ) -> None:
        """
        Helper used by initialization modules.
        """
        with torch.no_grad():
            self.center_raw.copy_(center_raw)
            self.scale_raw.copy_(scale_raw)
            self.angle_raw.copy_(angle_raw)
            self.alpha_raw.copy_(alpha_raw)
            self.color_raw.copy_(color_raw)
