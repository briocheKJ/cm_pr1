from __future__ import annotations

from models import Gaussian2DModel


class CustomInitializerTemplate:
    """
    TODO: implement your own initialization strategy here.

    Suggested steps:
    1. Decide how to initialize centers.
    2. Decide how to initialize sigma values.
    3. Decide how to initialize colors.
    4. Convert those values to raw parameters.
    5. Write them into the model with model.set_raw_parameters(...).
    """

    def __init__(self, config) -> None:
        self.config = config

    def initialize(self, model: Gaussian2DModel, target_image=None) -> None:
        # TODO: decide whether to use target_image or ignore it.
        # target_image, when provided, is an [H, W, 3] tensor in [0, 1].
        # Students can use it for image-aware initialization, or ignore it and
        # implement a target-independent initializer.
        raise NotImplementedError("TODO: implement your own initialization strategy here.")
