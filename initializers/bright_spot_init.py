# ============================================================
# [STUDENT FILE] 你需要在此文件中实现亮度感知初始化策略
#
# 思路：将高斯中心放在目标图像亮度最高的区域。
# ============================================================
from __future__ import annotations

import torch

from config import Config
from models import Gaussian2DModel, inverse_sigmoid, inverse_softplus


class BrightSpotGaussianInitializer:
    """
    Image-aware initialization based on image brightness.

    Strategy:
    - compute grayscale brightness of target image
    - select the brightest pixels as candidate Gaussian centers
    - initialize colors from those locations

    TODO: implement the initialize() method below.

    Available tools:
    - target_image.mean(dim=-1): compute grayscale brightness
    - torch.topk(...): find the top-k brightest pixels
    - model.set_raw_parameters(...): write raw parameters into the model
    - self.config.initializer.bright_spot: hyperparameters
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def initialize(self, model: Gaussian2DModel, target_image: torch.Tensor | None = None) -> None:
        """
        TODO: implement bright-spot initialization.

        Steps:
        1. Compute brightness = target_image.mean(dim=-1)
        2. Find the top num_gaussians brightest pixel locations
        3. Convert pixel indices to normalized [0,1] center coordinates
        4. Sample colors from target_image at those locations
        5. Set all raw parameters via model.set_raw_parameters(...)

        Hint: look at initializers/random_init.py for the parameter format.
        """
        raise NotImplementedError(
            "TODO: implement bright_spot initialization. "
            "See the docstring above for the strategy."
        )
