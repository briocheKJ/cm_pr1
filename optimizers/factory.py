from __future__ import annotations

from config import OptimizerConfig
from mode import is_teacher
from models import Gaussian2DModel
from optimizers.torch_baselines import build_torch_adam


def _get_base_lr(config: OptimizerConfig) -> float:
    """Extract the base learning rate for the selected optimizer."""
    name = config.name
    if name == "torch_adam":
        return config.torch_adam.lr
    if name == "student_sgd":
        return config.student_sgd.lr
    if name == "student_momentum":
        return config.student_momentum.lr
    if name == "student_adam":
        return config.student_adam.lr
    if name == "student_adamw":
        return config.student_adamw.lr
    if name == "student_muon":
        return config.student_muon.lr
    if name == "student_newton":
        return config.student_newton.lr
    raise ValueError(f"Unknown optimizer name: {name}")


def build_optimizer(model: Gaussian2DModel, config: OptimizerConfig):
    """
    Create an optimizer from a short name.

    The model's parameters are split into groups with per-group learning rates
    based on config.param_groups.
    """
    name = config.name
    base_lr = _get_base_lr(config)
    param_groups = model.get_param_groups(base_lr, config.param_groups)

    if name == "torch_adam":
        return build_torch_adam(param_groups=param_groups, config=config.torch_adam)

    # --- student_sgd ---
    if name == "student_sgd":
        if is_teacher():
            from _teacher_solutions.student_sgd import StudentSGD
        else:
            from optimizers.student_sgd import StudentSGD
        return StudentSGD(param_groups=param_groups)

    # --- student_momentum ---
    if name == "student_momentum":
        if is_teacher():
            from _teacher_solutions.student_momentum import StudentMomentum
        else:
            from optimizers.student_momentum import StudentMomentum
        return StudentMomentum(
            param_groups=param_groups,
            momentum=config.student_momentum.momentum,
        )

    # --- student_adam ---
    if name == "student_adam":
        if is_teacher():
            from _teacher_solutions.student_adam import StudentAdam
        else:
            from optimizers.student_adam import StudentAdam
        return StudentAdam(
            param_groups=param_groups,
            beta1=config.student_adam.beta1,
            beta2=config.student_adam.beta2,
            eps=config.student_adam.eps,
        )

    # --- student_adamw ---
    if name == "student_adamw":
        if is_teacher():
            from _teacher_solutions.student_adamw import StudentAdamW
        else:
            from optimizers.student_adamw import StudentAdamW
        return StudentAdamW(
            param_groups=param_groups,
            beta1=config.student_adamw.beta1,
            beta2=config.student_adamw.beta2,
            eps=config.student_adamw.eps,
            weight_decay=config.student_adamw.weight_decay,
        )

    # --- student_muon ---
    if name == "student_muon":
        if is_teacher():
            from _teacher_solutions.student_muon import StudentMuon
        else:
            from optimizers.student_muon import StudentMuon
        return StudentMuon(
            param_groups=param_groups,
            momentum=config.student_muon.momentum,
            weight_decay=config.student_muon.weight_decay,
            ns_steps=config.student_muon.ns_steps,
            nesterov=config.student_muon.nesterov,
        )

    # --- student_newton ---
    if name == "student_newton":
        if is_teacher():
            from _teacher_solutions.student_newton import StudentNewton
        else:
            from optimizers.student_newton import StudentNewton
        return StudentNewton(
            param_groups=param_groups,
            damping=config.student_newton.damping,
            max_curvature=config.student_newton.max_curvature,
        )

    raise ValueError(f"Unknown optimizer name: {name}")
