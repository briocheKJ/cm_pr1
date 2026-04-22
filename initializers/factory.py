from __future__ import annotations

from config import Config
from mode import is_teacher
from initializers.random_init import RandomGaussianInitializer
from initializers.grid_init import GridGaussianInitializer


def build_initializer(config: Config):
    """
    Create an initialization strategy from a short name.

    Supported:
    - random       (provided baseline)
    - grid         (provided)
    - image_sample (student TODO)
    - bright_spot  (student TODO)
    """
    name = config.initializer.name

    if name == "random":
        return RandomGaussianInitializer(config)

    if name == "grid":
        return GridGaussianInitializer(config)

    if name == "image_sample":
        if is_teacher():
            from _teacher_solutions.image_sample_init import ImageSampleGaussianInitializer
        else:
            from initializers.image_sample_init import ImageSampleGaussianInitializer
        return ImageSampleGaussianInitializer(config)

    if name == "bright_spot":
        if is_teacher():
            from _teacher_solutions.bright_spot_init import BrightSpotGaussianInitializer
        else:
            from initializers.bright_spot_init import BrightSpotGaussianInitializer
        return BrightSpotGaussianInitializer(config)

    raise ValueError(f"Unknown initializer name: {name}")
