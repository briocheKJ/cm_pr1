from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from config import Config


@dataclass
class TargetGaussianSpec:
    x: float
    y: float
    sigma_x: float
    sigma_y: float
    theta: float
    alpha: float
    r: float
    g: float
    b: float


class TxtGaussianTargetGenerator:
    """
    Generate a target image by reading Gaussian parameters from a text file.

    Expected format:
    - One Gaussian per line
    - Ignore blank lines
    - Ignore comment lines starting with '#'
    - Each valid line can use one of these formats:

      x  y  sigma  r  g  b
      x  y  sigma  alpha  r  g  b
      x  y  sigma_x  sigma_y  theta  r  g  b
      x  y  sigma_x  sigma_y  theta  alpha  r  g  b

    This keeps the simple isotropic case easy, while still allowing students to
    create anisotropic or semi-transparent debugging targets.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def generate(self, project_root: Path, device: torch.device) -> torch.Tensor:
        txt_path = project_root / self.config.target.gaussian_txt_path
        specs = self._read_specs(txt_path)
        return self._render_specs(specs=specs, device=device)

    def _read_specs(self, txt_path: Path) -> list[TargetGaussianSpec]:
        if not txt_path.exists():
            raise FileNotFoundError(f"Gaussian target txt file not found: {txt_path}")

        specs: list[TargetGaussianSpec] = []
        for line_idx, raw_line in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            values = [float(part) for part in parts]
            specs.append(self._parse_values(values=values, line_idx=line_idx, txt_path=txt_path))

        if not specs:
            raise ValueError(f"No valid Gaussian specs found in {txt_path}")

        return specs

    def _parse_values(self, values: list[float], line_idx: int, txt_path: Path) -> TargetGaussianSpec:
        if len(values) == 6:
            x, y, sigma, r, g, b = values
            return TargetGaussianSpec(x, y, sigma, sigma, 0.0, 1.0, r, g, b)

        if len(values) == 7:
            x, y, sigma, alpha, r, g, b = values
            return TargetGaussianSpec(x, y, sigma, sigma, 0.0, alpha, r, g, b)

        if len(values) == 8:
            x, y, sigma_x, sigma_y, theta, r, g, b = values
            return TargetGaussianSpec(x, y, sigma_x, sigma_y, theta, 1.0, r, g, b)

        if len(values) == 9:
            x, y, sigma_x, sigma_y, theta, alpha, r, g, b = values
            return TargetGaussianSpec(x, y, sigma_x, sigma_y, theta, alpha, r, g, b)

        raise ValueError(
            f"Line {line_idx} in {txt_path} must contain 6, 7, 8, or 9 values. "
            "See the txt target format in README.md."
        )

    def _render_specs(self, specs: list[TargetGaussianSpec], device: torch.device) -> torch.Tensor:
        centers = torch.tensor([[spec.x, spec.y] for spec in specs], dtype=torch.float32, device=device)
        scales = torch.tensor([[spec.sigma_x, spec.sigma_y] for spec in specs], dtype=torch.float32, device=device)
        angles = torch.tensor([[spec.theta] for spec in specs], dtype=torch.float32, device=device)
        alphas = torch.tensor([[spec.alpha] for spec in specs], dtype=torch.float32, device=device)
        colors = torch.tensor([[spec.r, spec.g, spec.b] for spec in specs], dtype=torch.float32, device=device)

        grid = self._build_pixel_grid(device=device)  # [H, W, 2]
        bg_color = torch.tensor(self.config.render.bg_color, dtype=torch.float32, device=device)

        diff = grid.unsqueeze(0) - centers[:, None, None, :]  # [N, H, W, 2]

        cos_theta = torch.cos(angles)[:, None, None, 0]  # [N, 1, 1]
        sin_theta = torch.sin(angles)[:, None, None, 0]  # [N, 1, 1]
        local_x = cos_theta * diff[..., 0] + sin_theta * diff[..., 1]   # [N, H, W]
        local_y = -sin_theta * diff[..., 0] + cos_theta * diff[..., 1]  # [N, H, W]

        sigma_x2 = scales[:, None, None, 0] ** 2
        sigma_y2 = scales[:, None, None, 1] ** 2
        mahalanobis = (local_x * local_x) / sigma_x2 + (local_y * local_y) / sigma_y2
        base_weights = torch.exp(-0.5 * mahalanobis)
        weights = alphas[:, None, None, 0] * base_weights

        weighted_rgb = torch.einsum("nhw,nc->hwc", weights, colors)
        weight_sum = weights.sum(dim=0).unsqueeze(-1)

        image = (weighted_rgb + self.config.render.eps * bg_color.view(1, 1, 3)) / (
            weight_sum + self.config.render.eps
        )
        return image.clamp(0.0, 1.0)

    def _build_pixel_grid(self, device: torch.device) -> torch.Tensor:
        coords = torch.linspace(0.0, 1.0, self.config.target.image_size, device=device)
        ys, xs = torch.meshgrid(coords, coords, indexing="ij")
        return torch.stack([xs, ys], dim=-1)
