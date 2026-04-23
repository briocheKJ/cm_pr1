from __future__ import annotations

import torch

from models import GaussianRenderParams


class GaussianRenderer:
    """
    Differentiable renderer for 2D Gaussians.
    """

    def __init__(
        self,
        image_size: int,
        bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        use_anisotropic: bool = False,
        use_alpha: bool = False,
    ) -> None:
        self.image_size = image_size
        self.bg_color = bg_color
        self.use_anisotropic = use_anisotropic
        self.use_alpha = use_alpha

    def render(self, params: GaussianRenderParams) -> torch.Tensor:
        centers = params.centers
        scales = params.scales
        rotations = params.rotations
        alphas = params.alphas
        colors = params.colors

        device = centers.device
        grid = self._build_pixel_grid(device=device)  # [H, W, 2]
        bg_color = torch.tensor(self.bg_color, dtype=centers.dtype, device=device)

        # Compare every Gaussian against every pixel.
        # diff shape: [N, H, W, 2]
        diff = grid.unsqueeze(0) - centers[:, None, None, :]

        if self.use_anisotropic:
            cos_theta = rotations[:, None, None, 0]
            sin_theta = rotations[:, None, None, 1]
            local_x = cos_theta * diff[..., 0] + sin_theta * diff[..., 1]
            local_y = -sin_theta * diff[..., 0] + cos_theta * diff[..., 1]

            sigma_x2 = scales[:, None, None, 0] ** 2
            sigma_y2 = scales[:, None, None, 1] ** 2
            mahalanobis = (local_x * local_x) / sigma_x2 + (local_y * local_y) / sigma_y2
            base_weights = torch.exp(-0.5 * mahalanobis)
        else:
            dist2 = (diff * diff).sum(dim=-1)  # [N, H, W]
            sigma2 = scales[:, None, None, 0] ** 2
            base_weights = torch.exp(-dist2 / (2.0 * sigma2))

        if self.use_alpha:
            weights = alphas[:, None, None, 0] * base_weights
        else:
            weights = base_weights

        weighted_rgb = torch.einsum("nhw,nc->hwc", weights, colors)  # [H, W, 3]

        image = bg_color.view(1, 1, 3) + weighted_rgb

        return image.clamp(0.0, 1.0)

    def _build_pixel_grid(self, device: torch.device) -> torch.Tensor:
        coords = torch.linspace(0.0, 1.0, self.image_size, device=device)
        ys, xs = torch.meshgrid(coords, coords, indexing="ij")
        return torch.stack([xs, ys], dim=-1)
