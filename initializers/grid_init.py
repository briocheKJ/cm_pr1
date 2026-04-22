from __future__ import annotations

import math

import torch

from config import Config
from models import Gaussian2DModel, inverse_sigmoid, inverse_softplus


class GridGaussianInitializer:
    """
    Place Gaussian centers on a coarse grid.

    This is useful for experiments where students want a more structured start.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def initialize(self, model: Gaussian2DModel, target_image: torch.Tensor | None = None) -> None:
        del target_image
        num_gaussians = model.num_gaussians
        device = model.center_raw.device
        cfg = self.config.initializer.grid

        grid_size = math.ceil(math.sqrt(num_gaussians))
        coords = torch.linspace(cfg.margin_min, cfg.margin_max, grid_size, device=device)
        ys, xs = torch.meshgrid(coords, coords, indexing="ij")
        centers = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)[:num_gaussians]

        sigma_value = cfg.sigma_scale / grid_size
        sigmas = torch.full((num_gaussians, 1), sigma_value, device=device)
        scales = sigmas.repeat(1, 2)
        angles = torch.zeros(num_gaussians, 1, device=device)
        alphas = torch.full((num_gaussians, 1), cfg.alpha, device=device)

        colors = torch.full((num_gaussians, 3), cfg.color_mean, device=device)
        colors = colors + cfg.color_std * torch.randn_like(colors)
        colors = colors.clamp(cfg.color_min, cfg.color_max)

        model.set_raw_parameters(
            center_raw=inverse_sigmoid(centers),
            scale_raw=inverse_softplus(scales),
            angle_raw=angles,
            alpha_raw=inverse_sigmoid(alphas),
            color_raw=inverse_sigmoid(colors),
        )
