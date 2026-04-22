from __future__ import annotations

from pathlib import Path

from config import Config
from train import train


def run_smoke_test() -> None:
    output_dir = "outputs_smoke"
    config = Config()
    config.target.image_size = 64
    config.target.name = "synthetic_shapes"
    config.train.num_steps = 5
    config.train.save_every = 5
    config.train.print_every = 1
    config.system.output_dir = output_dir
    train(config)

    project_root = Path(__file__).resolve().parent
    expected_files = [
        project_root / output_dir / "target.png",
        project_root / output_dir / "reconstruction_final.png",
        project_root / output_dir / "loss_curve.png",
        project_root / output_dir / "recon_step_0005.png",
        project_root / output_dir / "comparison.png",
        project_root / output_dir / "metrics.txt",
    ]

    for path in expected_files:
        assert path.exists(), f"Missing expected smoke test output: {path}"

    print("Smoke test passed.")


if __name__ == "__main__":
    run_smoke_test()
