from __future__ import annotations

from config import Config
from target_generators.custom_target_generator_template import CustomTargetGeneratorTemplate
from target_generators.image_target_generator import ImageTargetGenerator
from target_generators.synthetic_shapes_generator import SyntheticShapesTargetGenerator
from target_generators.txt_gaussian_generator import TxtGaussianTargetGenerator


def build_target_generator(config: Config):
    """
    Create a target image generator from a short name.

    Supported today:
    - image
    - synthetic_shapes
    - txt_gaussians

    Reserved for future student extensions:
    - custom
    """
    name = config.target.name

    if name == "image":
        return ImageTargetGenerator(config)

    if name == "synthetic_shapes":
        return SyntheticShapesTargetGenerator(config)

    if name == "txt_gaussians":
        return TxtGaussianTargetGenerator(config)

    if name == "custom":
        return CustomTargetGeneratorTemplate(config)

    raise ValueError(f"Unknown target generator name: {name}")
