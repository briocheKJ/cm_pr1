from __future__ import annotations

from collections.abc import Iterable

import torch

from config import OptimizerConfig
from mode import is_teacher
from optimizers.torch_baselines import build_torch_adam


def build_optimizer(
    params: Iterable[torch.nn.Parameter],
    config: OptimizerConfig,
):
    """
    Create an optimizer from a short name.

    Supported today:
    - torch_adam       (provided baseline)
    - student_sgd      (student TODO)
    - student_adam     (student TODO)
    - student_adamw    (student TODO)
    - student_muon     (student TODO, bonus)
    - student_newton   (student TODO, bonus)
    """
    name = config.name

    if name == "torch_adam":
        return build_torch_adam(params=params, config=config.torch_adam)

    # --- student_sgd ---
    if name == "student_sgd":
        if is_teacher():
            from _teacher_solutions.student_sgd import StudentSGD
        else:
            from optimizers.student_sgd import StudentSGD
        return StudentSGD(params=params, lr=config.student_sgd.lr)

    # --- student_adam ---
    if name == "student_adam":
        if is_teacher():
            from _teacher_solutions.student_adam import StudentAdam
        else:
            from optimizers.student_adam import StudentAdam
        return StudentAdam(
            params=params,
            lr=config.student_adam.lr,
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
            params=params,
            lr=config.student_adamw.lr,
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
            params=params,
            lr=config.student_muon.lr,
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
            params=params,
            lr=config.student_newton.lr,
            damping=config.student_newton.damping,
            max_curvature=config.student_newton.max_curvature,
        )

    if name == "student_momentum":
        raise NotImplementedError(
            f"Optimizer '{name}' is reserved for student implementations. "
            "Add the implementation under optimizers/ and register it here."
        )

    raise ValueError(f"Unknown optimizer name: {name}")
