from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from utils import save_image_panel


@dataclass
class EvalResult:
    mse: float
    mae: float
    psnr: float


def evaluate_prediction(prediction: torch.Tensor, target: torch.Tensor) -> EvalResult:
    """
    Compute a few simple image reconstruction metrics.

    These numbers are intentionally easy to read:
    - lower MSE is better
    - lower MAE is better
    - higher PSNR is better
    """
    mse = torch.mean((prediction - target) ** 2).item()
    mae = torch.mean(torch.abs(prediction - target)).item()
    psnr = -10.0 * torch.log10(torch.tensor(mse + 1e-12)).item()
    return EvalResult(mse=mse, mae=mae, psnr=psnr)


def save_evaluation_report(
    result: EvalResult,
    path: str | Path,
    optimizer_name: str,
    init_name: str,
    num_steps: int,
    image_size: int,
) -> None:
    """
    Save a small text report so students can compare runs directly.
    """
    path = Path(path)
    lines = [
        "2DGS Evaluation Summary",
        f"optimizer: {optimizer_name}",
        f"initializer: {init_name}",
        f"num_steps: {num_steps}",
        f"image_size: {image_size}",
        f"mse: {result.mse:.8f}",
        f"mae: {result.mae:.8f}",
        f"psnr: {result.psnr:.4f} dB",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_comparison_visual(
    target: torch.Tensor,
    prediction: torch.Tensor,
    path: str | Path,
) -> None:
    """
    Save a side-by-side comparison figure:
    target / prediction / absolute error.
    """
    error = torch.abs(prediction - target)
    save_image_panel(
        images=[target, prediction, error],
        titles=["Target", "Prediction", "Absolute Error"],
        path=path,
    )
