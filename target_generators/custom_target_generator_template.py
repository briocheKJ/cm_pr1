from __future__ import annotations

from pathlib import Path

import torch


class CustomTargetGeneratorTemplate:
    """
    TODO: implement your own target image generator here.

    Suggested steps:
    1. Decide how you want to describe the target image.
    2. Create a tensor of shape [H, W, 3] with values in [0, 1].
    3. Return that tensor in generate(...).
    4. Register your generator in target_generators/factory.py.

    This is useful for building debugging targets for your own experiments.
    """

    def __init__(self, config) -> None:
        self.config = config

    def generate(self, project_root: Path, device: torch.device) -> torch.Tensor:
        del project_root, device
        raise NotImplementedError("TODO: implement your own target image generator here.")
