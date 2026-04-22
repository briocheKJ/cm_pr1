"""
Competition evaluator for Experiment 1.

Usage:
    python experiments/run_competition.py --config experiments/my_competition_config.py

This script:
1. Loads the student's config file (must define get_sprint_config / get_standard_config)
2. Runs both tracks on all 5 test images
3. Runs both tracks on all 10 test images (5 real RGB + 5 txt synthetic)
4. Reports per-image PSNR and the average PSNR for each track
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# Ensure the project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import Config
from evaluation import evaluate_prediction
from initializers import build_initializer
from losses import build_loss
from models import Gaussian2DModel
from optimizers import build_optimizer
from renderer import GaussianRenderer
from target_generators import build_target_generator
from mode import set_mode
from utils import ensure_dir, resolve_device, save_image, set_seed


# ---------------------------------------------------------------------------
# Test image definitions
# ---------------------------------------------------------------------------

TEST_IMAGES: list[dict] = [
    # --- 5 real RGB images ---
    {
        "name": "R1_starry_night",
        "target_name": "image",
        "image_path": "data/Starry_Night_256.png",
        "txt_path": None,
    },
    {
        "name": "R2_blackswan",
        "target_name": "image",
        "image_path": "data/competition/blackswan_256.png",
        "txt_path": None,
    },
    {
        "name": "R3_flamingo",
        "target_name": "image",
        "image_path": "data/competition/flamingo_256.png",
        "txt_path": None,
    },
    {
        "name": "R4_car_roundabout",
        "target_name": "image",
        "image_path": "data/competition/car-roundabout_256.png",
        "txt_path": None,
    },
    {
        "name": "R5_parkour",
        "target_name": "image",
        "image_path": "data/competition/parkour_256.png",
        "txt_path": None,
    },
    # --- 5 txt synthetic Gaussian targets ---
    {
        "name": "S1_sparse_colorful",
        "target_name": "txt_gaussians",
        "image_path": None,
        "txt_path": "data/competition/t3_sparse_colorful.txt",
    },
    {
        "name": "S2_dense_cluster",
        "target_name": "txt_gaussians",
        "image_path": None,
        "txt_path": "data/competition/t4_dense_cluster.txt",
    },
    {
        "name": "S3_anisotropic_mix",
        "target_name": "txt_gaussians",
        "image_path": None,
        "txt_path": "data/competition/t5_anisotropic_mix.txt",
    },
    {
        "name": "S4_ten_translucent",
        "target_name": "txt_gaussians",
        "image_path": None,
        "txt_path": "data/examples/04_ten_translucent_stars.txt",
    },
    {
        "name": "S5_ten_colorful",
        "target_name": "txt_gaussians",
        "image_path": None,
        "txt_path": "data/examples/05_ten_colorful_stars.txt",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_student_module(config_path: str) -> ModuleType:
    """Dynamically import the student's config file."""
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    spec = importlib.util.spec_from_file_location("student_config", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_hard_constraints(config: Config, track: str, num_steps: int) -> None:
    """Ensure the student did not modify hard constraints."""
    errors: list[str] = []
    if config.model.num_gaussians != 200:
        errors.append(f"num_gaussians must be 200, got {config.model.num_gaussians}")
    if config.render.bg_color != (0.0, 0.0, 0.0):
        errors.append(f"bg_color must be (0,0,0), got {config.render.bg_color}")
    if config.system.seed != 42:
        errors.append(f"seed must be 42, got {config.system.seed}")
    if config.target.image_size != 256:
        errors.append(f"image_size must be 256, got {config.target.image_size}")
    if config.train.num_steps != num_steps:
        errors.append(f"num_steps must be {num_steps} for {track}, got {config.train.num_steps}")
    if errors:
        raise ValueError(f"Hard constraint violation in {track} config:\n" + "\n".join(errors))


def run_single(config: Config, test_image: dict, output_dir: Path) -> float:
    """Run training on one test image, return PSNR."""
    set_seed(config.system.seed)
    device = resolve_device(config.system.device)

    # Override target settings per test image.
    config.target.name = test_image["target_name"]
    if test_image["image_path"] is not None:
        config.target.image_path = test_image["image_path"]
    if test_image["txt_path"] is not None:
        config.target.gaussian_txt_path = test_image["txt_path"]

    # Disable video export for speed.
    config.visualization.save_video = False
    config.system.output_dir = str(output_dir)

    target_generator = build_target_generator(config)
    target = target_generator.generate(project_root=PROJECT_ROOT, device=device)

    model = Gaussian2DModel(num_gaussians=config.model.num_gaussians).to(device)
    initializer = build_initializer(config)
    initializer.initialize(model, target_image=target)

    renderer = GaussianRenderer(
        image_size=config.target.image_size,
        bg_color=config.render.bg_color,
        eps=config.render.eps,
        use_anisotropic=config.model.use_anisotropic,
        use_alpha=config.model.use_alpha,
    )
    optimizer = build_optimizer(params=model.parameters(), config=config.optimizer)
    loss_fn = build_loss(config.loss)

    for step in range(1, config.train.num_steps + 1):
        optimizer.zero_grad()
        render_params = model.get_render_params()
        prediction = renderer.render(render_params)
        loss = loss_fn(prediction, target)
        loss.backward()
        optimizer.step()

    # Final evaluation.
    with torch.no_grad():
        render_params = model.get_render_params()
        prediction = renderer.render(render_params)

    result = evaluate_prediction(prediction=prediction, target=target)

    # Save artifacts.
    out = ensure_dir(output_dir)
    save_image(target, out / "target.png")
    save_image(prediction, out / "prediction.png")

    return result.psnr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_track(get_config_fn, track_name: str, num_steps: int, base_output: Path) -> float:
    """Run one track across all test images, return average PSNR."""
    print(f"\n{'=' * 60}")
    print(f"  Track: {track_name} ({num_steps} steps)")
    print(f"{'=' * 60}")

    psnrs: list[float] = []
    for test_image in TEST_IMAGES:
        config = get_config_fn()
        validate_hard_constraints(config, track_name, num_steps)

        out_dir = base_output / track_name / test_image["name"]
        psnr = run_single(config, test_image, out_dir)
        psnrs.append(psnr)
        print(f"  {test_image['name']:30s}  PSNR = {psnr:.4f} dB")

    avg_psnr = sum(psnrs) / len(psnrs)
    print(f"  {'AVERAGE':30s}  PSNR = {avg_psnr:.4f} dB")
    return avg_psnr


def main() -> None:
    parser = argparse.ArgumentParser(description="Competition evaluator for Experiment 1")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to student config file (must define get_sprint_config and get_standard_config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs_competition",
        help="Output directory (default: outputs_competition)",
    )
    parser.add_argument(
        "--mode",
        choices=["student", "teacher"],
        default="student",
        help="student: uses student-implemented files; teacher: loads reference solutions",
    )
    args = parser.parse_args()

    set_mode(args.mode)

    student_module = load_student_module(args.config)
    base_output = ensure_dir(PROJECT_ROOT / args.output)

    if not hasattr(student_module, "get_sprint_config"):
        raise AttributeError("Config file must define get_sprint_config()")
    if not hasattr(student_module, "get_standard_config"):
        raise AttributeError("Config file must define get_standard_config()")

    sprint_avg = run_track(student_module.get_sprint_config, "sprint", 100, base_output)
    standard_avg = run_track(student_module.get_standard_config, "standard", 500, base_output)

    print(f"\n{'=' * 60}")
    print(f"  FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"  Sprint  (100 steps)  avg PSNR = {sprint_avg:.4f} dB")
    print(f"  Standard (500 steps) avg PSNR = {standard_avg:.4f} dB")
    print(f"{'=' * 60}")

    # Save summary.
    summary_path = base_output / "competition_summary.txt"
    summary_path.write_text(
        f"Sprint  avg PSNR: {sprint_avg:.4f} dB\n"
        f"Standard avg PSNR: {standard_avg:.4f} dB\n",
        encoding="utf-8",
    )
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
