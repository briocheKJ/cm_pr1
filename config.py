from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SystemConfig:
    """
    Settings that control reproducibility, device choice, and where outputs go.
    """

    seed: int = 42
    device: str = "auto"
    output_dir: str = "outputs"


@dataclass
class TargetConfig:
    """
    Settings for how the target image is created.
    """

    name: str = "image"
    image_size: int = 256
    image_path: str = "data/Starry_Night_256.png"
    gaussian_txt_path: str = "data/sample_two_gaussians.txt"


@dataclass
class ModelConfig:
    """
    Settings for the trainable Gaussian set.
    """

    num_gaussians: int = 100
    use_anisotropic: bool = True
    use_alpha: bool = True


@dataclass
class RenderConfig:
    """
    Settings used by the differentiable renderer.
    """

    bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    eps: float = 1e-6


@dataclass
class TrainLoopConfig:
    """
    Settings for the training loop itself.
    """

    num_steps: int = 200
    print_every: int = 20
    save_every: int = 50


@dataclass
class LossConfig:
    """
    Settings for the training loss.
    """

    name: str = "mse"
    l1_weight: float = 0.2
    edge_weight: float = 0.1
    charbonnier_eps: float = 1e-3


# ---------------------------------------------------------------------------
# Parameter group learning rate multipliers
# ---------------------------------------------------------------------------

@dataclass
class ParamGroupConfig:
    """
    Per-parameter-group learning rate multipliers.

    The effective lr for each parameter group is:
        effective_lr = base_lr * multiplier

    For example, setting center_lr_scale=2.0 means the center parameters
    learn at twice the base learning rate.
    """

    center_lr_scale: float = 1.0
    scale_lr_scale: float = 1.0
    angle_lr_scale: float = 1.0
    alpha_lr_scale: float = 1.0
    color_lr_scale: float = 1.0


# ---------------------------------------------------------------------------
# Learning rate scheduler
# ---------------------------------------------------------------------------

@dataclass
class SchedulerConfig:
    """
    Learning rate scheduler settings.

    The scheduler returns a multiplier in [min_lr_scale, 1.0] applied to all
    parameter group learning rates each step.
    """

    name: str = "constant"
    min_lr_scale: float = 0.01
    warmup_steps: int = 0
    step_size: int = 100
    gamma: float = 0.5


# ---------------------------------------------------------------------------
# Optimizer configs
# ---------------------------------------------------------------------------

@dataclass
class TorchAdamConfig:
    lr: float = 5e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass
class StudentSGDConfig:
    lr: float = 5e-2


@dataclass
class StudentMomentumConfig:
    lr: float = 5e-2
    momentum: float = 0.9


@dataclass
class StudentAdamConfig:
    lr: float = 5e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass
class StudentAdamWConfig:
    lr: float = 5e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-2


@dataclass
class StudentMuonConfig:
    lr: float = 5e-2
    momentum: float = 0.95
    weight_decay: float = 0.0
    ns_steps: int = 5
    nesterov: bool = True


@dataclass
class StudentNewtonConfig:
    lr: float = 5e-2
    damping: float = 1e-3
    max_curvature: float = 1e3


@dataclass
class OptimizerConfig:
    """
    All optimizer-related settings live here.

    Students should usually only change:
    - name
    - the sub-config that matches that name
    - param_groups (per-parameter learning rate multipliers)
    """

    name: str = "torch_adam"
    param_groups: ParamGroupConfig = field(default_factory=ParamGroupConfig)
    torch_adam: TorchAdamConfig = field(default_factory=TorchAdamConfig)
    student_sgd: StudentSGDConfig = field(default_factory=StudentSGDConfig)
    student_momentum: StudentMomentumConfig = field(default_factory=StudentMomentumConfig)
    student_adam: StudentAdamConfig = field(default_factory=StudentAdamConfig)
    student_adamw: StudentAdamWConfig = field(default_factory=StudentAdamWConfig)
    student_muon: StudentMuonConfig = field(default_factory=StudentMuonConfig)
    student_newton: StudentNewtonConfig = field(default_factory=StudentNewtonConfig)


# ---------------------------------------------------------------------------
# Initializer configs
# ---------------------------------------------------------------------------

@dataclass
class RandomInitConfig:
    center_min: float = 0.1
    center_max: float = 0.9
    sigma_min: float = 0.10
    sigma_max: float = 0.14
    alpha: float = 0.95
    color_mean: float = 0.5
    color_std: float = 0.05
    color_min: float = 0.05
    color_max: float = 0.95


@dataclass
class GridInitConfig:
    margin_min: float = 0.15
    margin_max: float = 0.85
    sigma_scale: float = 0.35
    alpha: float = 0.95
    color_mean: float = 0.5
    color_std: float = 0.02
    color_min: float = 0.05
    color_max: float = 0.95


@dataclass
class ImageSampleInitConfig:
    center_min: float = 0.1
    center_max: float = 0.9
    sigma_min: float = 0.08
    sigma_max: float = 0.11
    alpha: float = 0.95


@dataclass
class BrightSpotInitConfig:
    fallback_center_min: float = 0.1
    fallback_center_max: float = 0.9
    sigma_bias: float = 0.05
    sigma_brightness_scale: float = 0.08
    alpha_bias: float = 0.6
    alpha_brightness_scale: float = 0.35


@dataclass
class InitializerConfig:
    """
    All initialization-related settings live here.

    The initializer interface supports:
    - target-independent methods
    - target-aware methods that inspect the target image
    """

    name: str = "random"
    random: RandomInitConfig = field(default_factory=RandomInitConfig)
    grid: GridInitConfig = field(default_factory=GridInitConfig)
    image_sample: ImageSampleInitConfig = field(default_factory=ImageSampleInitConfig)
    bright_spot: BrightSpotInitConfig = field(default_factory=BrightSpotInitConfig)


@dataclass
class EvaluationConfig:
    """
    Settings for saved evaluation artifacts.
    """

    metrics_filename: str = "metrics.txt"
    comparison_filename: str = "comparison.png"
    loss_curve_filename: str = "loss_curve.png"
    final_reconstruction_filename: str = "reconstruction_final.png"
    target_filename: str = "target.png"


@dataclass
class VisualizationConfig:
    """
    Optional visualization settings.

    Video export is off by default so the baseline run stays lightweight.
    """

    save_video: bool = True
    video_every: int = 10
    video_filename: str = "optimization.mp4"
    video_frames_dir: str = "video_frames"


@dataclass
class Config:
    """
    Structured top-level config for the whole starter project.

    The main sections students will usually touch are:
    - target
    - model
    - optimizer
    - scheduler
    - initializer
    - loss
    - train
    """

    system: SystemConfig = field(default_factory=SystemConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    train: TrainLoopConfig = field(default_factory=TrainLoopConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    initializer: InitializerConfig = field(default_factory=InitializerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
