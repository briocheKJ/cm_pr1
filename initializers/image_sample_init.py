# ============================================================
# [STUDENT FILE] 你需要在此文件中实现图像采样初始化策略
#
# 思路：随机放置高斯中心，但从目标图像对应位置采样颜色。
# ============================================================
from __future__ import annotations

import torch

from config import Config
from models import Gaussian2DModel, inverse_sigmoid, inverse_softplus


class ImageSampleGaussianInitializer:
    """
    Image-aware initialization.

    Strategy:
    - place centers randomly
    - sample the target image color at those centers
    - initialize Gaussian colors from the sampled pixels

    TODO: implement the initialize() method below.

    Available tools:
    - model.num_gaussians: number of Gaussians
    - model.center_raw.device: the device to create tensors on
    - target_image: [H, W, 3] tensor in [0, 1], access pixel color via target_image[y, x]
    - model.set_raw_parameters(...): write raw parameters into the model
    - inverse_sigmoid(x): convert [0,1] value to raw parameter space
    - inverse_softplus(x): convert positive value to raw parameter space
    - self.config.initializer.image_sample: hyperparameters (center_min, center_max, sigma_min, sigma_max, alpha)
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def initialize(self, model: Gaussian2DModel, target_image: torch.Tensor | None = None) -> None:
        """
        TODO: implement image-sample initialization.

        Steps:
        1. Generate random centers in [center_min, center_max]
        2. Convert centers to pixel coordinates and sample colors from target_image
        3. Generate random sigma values
        4. Set all raw parameters via model.set_raw_parameters(...)

        Hint: look at initializers/random_init.py for the parameter format.
        """
        raise NotImplementedError(
            "TODO: implement image_sample initialization. "
            "See the docstring above for the strategy."
        )
