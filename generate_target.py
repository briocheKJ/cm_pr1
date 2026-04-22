from __future__ import annotations

from pathlib import Path

from config import Config
from target_generators import build_target_generator
from utils import ensure_dir, resolve_device, save_image, set_seed


def generate_target(config: Config, output_name: str = "generated_target.png") -> None:
    """
    Generate a target image without running training.

    This is useful when students want to debug a txt-defined Gaussian target
    image before testing their optimizer or initializer.
    """
    set_seed(config.system.seed)

    project_root = Path(__file__).resolve().parent
    output_dir = ensure_dir(project_root / config.system.output_dir)
    device = resolve_device(config.system.device)

    target_generator = build_target_generator(config)
    target = target_generator.generate(project_root=project_root, device=device)
    output_path = output_dir / output_name
    save_image(target, output_path)

    print(f"Target generator: {config.target.name}")
    print(f"Image size: {config.target.image_size}x{config.target.image_size}")
    print(f"Saved generated target to: {output_path}")


if __name__ == "__main__":
    generate_target(Config())
