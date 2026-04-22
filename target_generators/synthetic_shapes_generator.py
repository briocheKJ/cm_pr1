from __future__ import annotations

from pathlib import Path

import torch

from config import Config
from utils import make_synthetic_target_image


class SyntheticShapesTargetGenerator:
    """
    Programmatically generate a simple RGB target image.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def generate(self, project_root: Path, device: torch.device) -> torch.Tensor:
        del project_root
        return make_synthetic_target_image(image_size=self.config.target.image_size, device=device)
