"""
Competition config template.

Students: edit this file and submit it.
You must define two functions: get_sprint_config() and get_standard_config().
"""
from config import Config


def get_sprint_config() -> Config:
    """Sprint track: 100 steps, optimize for fast convergence."""
    config = Config()
    # === Hard constraints — DO NOT modify ===
    config.model.num_gaussians = 200
    config.render.bg_color = (0.0, 0.0, 0.0)
    config.system.seed = 42
    config.target.image_size = 256
    config.train.num_steps = 100
    # === Free to configure below ===
    config.loss.name = "mse"
    config.initializer.name = "random"
    config.optimizer.name = "torch_adam"
    config.optimizer.torch_adam.lr = 5e-2
    config.model.use_anisotropic = True
    config.model.use_alpha = True
    return config


def get_standard_config() -> Config:
    """Standard track: 500 steps, optimize for best PSNR."""
    config = Config()
    # === Hard constraints — DO NOT modify ===
    config.model.num_gaussians = 200
    config.render.bg_color = (0.0, 0.0, 0.0)
    config.system.seed = 42
    config.target.image_size = 256
    config.train.num_steps = 500
    # === Free to configure below ===
    config.loss.name = "mse"
    config.initializer.name = "random"
    config.optimizer.name = "torch_adam"
    config.optimizer.torch_adam.lr = 5e-2
    config.model.use_anisotropic = True
    config.model.use_alpha = True
    return config
