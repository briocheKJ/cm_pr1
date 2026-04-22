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

    # --- Custom initializer: try to import from student's custom module ---
    try:
        import importlib
        mod = importlib.import_module(f"initializers.{name}_init")
        # Look for a class ending with "Initializer"
        cls = None
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if isinstance(obj, type) and attr_name.endswith("Initializer") and attr_name != "Initializer":
                cls = obj
                break
        if cls is not None:
            return cls(config)
    except ModuleNotFoundError:
        pass

    raise ValueError(
        f"Unknown initializer name: {name}\n"
        f"If you defined a custom initializer, create a file `initializers/{name}_init.py`\n"
        f"with a class ending in `Initializer` that has an `initialize(model, target_image)` method."
    )
