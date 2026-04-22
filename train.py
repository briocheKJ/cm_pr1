from __future__ import annotations

from pathlib import Path

import torch

from config import Config
from evaluation import evaluate_prediction, save_comparison_visual, save_evaluation_report
from initializers import build_initializer
from losses import build_loss
from models import Gaussian2DModel
from optimizers import build_optimizer
from renderer import GaussianRenderer
from target_generators import build_target_generator
from utils import (
    build_animation_from_frames,
    ensure_dir,
    plot_loss_curve,
    resolve_device,
    save_image,
    save_training_frame,
    set_seed,
)


def train(config: Config) -> None:
    """
    Run the default 2DGS training loop.

    This function intentionally keeps the whole pipeline visible:
    target -> model -> renderer -> loss -> optimizer.
    """
    set_seed(config.system.seed)

    device = resolve_device(config.system.device)
    project_root = Path(__file__).resolve().parent
    output_dir = ensure_dir(project_root / config.system.output_dir)
    frame_paths: list[Path] = []
    frames_dir: Path | None = None
    if config.visualization.save_video:
        frames_dir = ensure_dir(output_dir / config.visualization.video_frames_dir)

    target_generator = build_target_generator(config)
    target = target_generator.generate(project_root=project_root, device=device)
    save_image(target, output_dir / config.evaluation.target_filename)

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
    optimizer = build_optimizer(
        params=model.parameters(),
        config=config.optimizer,
    )
    loss_fn = build_loss(config.loss)

    losses: list[float] = []

    print(f"Device: {device}")
    print(f"Image size: {config.target.image_size}x{config.target.image_size}")
    print(f"Number of Gaussians: {config.model.num_gaussians}")
    print(f"Use anisotropic: {config.model.use_anisotropic}")
    print(f"Use alpha: {config.model.use_alpha}")
    print(f"Target generator: {config.target.name}")
    print(f"Initializer: {config.initializer.name}")
    print(f"Optimizer: {config.optimizer.name}")
    print(f"Loss: {config.loss.name}")
    print(f"Save video: {config.visualization.save_video}")

    for step in range(1, config.train.num_steps + 1):
        optimizer.zero_grad()

        render_params = model.get_render_params()
        prediction = renderer.render(render_params)

        loss = loss_fn(prediction, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step == 1 or step % config.train.print_every == 0 or step == config.train.num_steps:
            print(f"step={step:04d}  loss={loss.item():.6f}")

        if step % config.train.save_every == 0 or step == config.train.num_steps:
            save_image(prediction, output_dir / f"recon_step_{step:04d}.png")

        if config.visualization.save_video and (
            step == 1
            or step % config.visualization.video_every == 0
            or step == config.train.num_steps
        ):
            assert frames_dir is not None
            frame_path = frames_dir / f"frame_{step:04d}.png"
            save_training_frame(
                target=target,
                prediction=prediction,
                path=frame_path,
                step=step,
            )
            frame_paths.append(frame_path)

    save_image(prediction, output_dir / config.evaluation.final_reconstruction_filename)
    plot_loss_curve(losses, output_dir / config.evaluation.loss_curve_filename)

    eval_result = evaluate_prediction(prediction=prediction, target=target)
    save_evaluation_report(
        result=eval_result,
        path=output_dir / config.evaluation.metrics_filename,
        optimizer_name=config.optimizer.name,
        init_name=config.initializer.name,
        num_steps=config.train.num_steps,
        image_size=config.target.image_size,
    )
    save_comparison_visual(
        target=target,
        prediction=prediction,
        path=output_dir / config.evaluation.comparison_filename,
    )

    if config.visualization.save_video:
        animation_path = build_animation_from_frames(
            frame_paths=frame_paths,
            output_path=output_dir / config.visualization.video_filename,
        )
        print(f"Saved optimization animation to: {animation_path}")

    print(
        "Final evaluation: "
        f"MSE={eval_result.mse:.8f}, "
        f"MAE={eval_result.mae:.8f}, "
        f"PSNR={eval_result.psnr:.4f} dB"
    )

    print(f"Saved outputs to: {output_dir}")
