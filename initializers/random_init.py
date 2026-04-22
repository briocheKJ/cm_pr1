from __future__ import annotations

import torch

from config import Config
from models import Gaussian2DModel, inverse_sigmoid, inverse_softplus


class RandomGaussianInitializer:
    """
    Stable random initialization used by the default baseline.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def initialize(self, model: Gaussian2DModel, target_image: torch.Tensor | None = None) -> None:
        del target_image
        num_gaussians = model.num_gaussians
        device = model.center_raw.device
        cfg = self.config.initializer.random

        center_init = cfg.center_min + (cfg.center_max - cfg.center_min) * torch.rand(num_gaussians, 2, device=device)
        sigma_init = cfg.sigma_min + (cfg.sigma_max - cfg.sigma_min) * torch.rand(num_gaussians, 1, device=device)
        scale_init = sigma_init.repeat(1, 2)
        angle_init = torch.zeros(num_gaussians, 1, device=device)
        alpha_init = torch.full((num_gaussians, 1), cfg.alpha, device=device)
        color_init = cfg.color_mean + cfg.color_std * torch.randn(num_gaussians, 3, device=device)
        color_init = color_init.clamp(cfg.color_min, cfg.color_max)

        model.set_raw_parameters(
            center_raw=inverse_sigmoid(center_init),
            scale_raw=inverse_softplus(scale_init),
            angle_raw=angle_init,
            alpha_raw=inverse_sigmoid(alpha_init),
            color_raw=inverse_sigmoid(color_init),
        )
