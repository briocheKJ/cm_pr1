from __future__ import annotations

from pathlib import Path

import torch

from config import Config
from utils import load_rgb_image, save_center_cropped_image


class ImageTargetGenerator:
    """
    Load a real image target and resize it to the configured square resolution.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def generate(self, project_root: Path, device: torch.device) -> torch.Tensor:
        target_path = project_root / self.config.target.image_path
        if not target_path.exists():
            raise FileNotFoundError(
                f"Target image not found: {target_path}\n"
                f"Please set config.target.image_path to a valid image path."
            )
        return load_rgb_image(target_path, image_size=self.config.target.image_size, device=device)
