# ============================================================
# [STUDENT FILE] 你可以在此文件中实现学习率调度器
#
# 已提供: constant (基线，不调度)
# 需实现: cosine, warmup_cosine, step_decay
#
# 调度器是一个函数: (step, total_steps) -> lr_multiplier
# 返回值在 [min_lr_scale, 1.0] 之间，乘以 base_lr 得到当前学习率。
# ============================================================
from __future__ import annotations

import math
from typing import Callable

from config import SchedulerConfig


def constant_schedule(step: int, total_steps: int) -> float:
    """Always returns 1.0 — no learning rate decay."""
    return 1.0


def cosine_schedule(step: int, total_steps: int, min_lr_scale: float = 0.01) -> float:
    """
    Cosine annealing: lr decays from 1.0 to min_lr_scale following a cosine curve.

    Formula:
        scale = min_lr_scale + 0.5 * (1 - min_lr_scale) * (1 + cos(pi * step / total_steps))

    TODO: implement this scheduler.
    """
    raise NotImplementedError("TODO: implement cosine_schedule")


def warmup_cosine_schedule(
    step: int, total_steps: int, warmup_steps: int, min_lr_scale: float = 0.01,
) -> float:
    """
    Linear warmup followed by cosine annealing.

    - Steps 1..warmup_steps: linearly increase from min_lr_scale to 1.0
    - Steps warmup_steps+1..total_steps: cosine decay from 1.0 to min_lr_scale

    TODO: implement this scheduler.
    """
    raise NotImplementedError("TODO: implement warmup_cosine_schedule")


def step_decay_schedule(
    step: int, total_steps: int, step_size: int = 100, gamma: float = 0.5, min_lr_scale: float = 0.01,
) -> float:
    """
    Step decay: multiply lr by gamma every step_size steps.

    Formula:
        scale = max(gamma ^ (step // step_size), min_lr_scale)

    TODO: implement this scheduler.
    """
    raise NotImplementedError("TODO: implement step_decay_schedule")


def build_scheduler(config: SchedulerConfig) -> Callable[[int, int], float]:
    """
    Build a scheduler function from config.

    Returns a callable (step, total_steps) -> lr_multiplier.
    """
    from mode import is_teacher

    name = config.name

    if name == "constant":
        return constant_schedule

    if is_teacher():
        from _teacher_solutions.schedulers import build_teacher_scheduler
        return build_teacher_scheduler(config)

    if name == "cosine":
        return lambda step, total: cosine_schedule(step, total, min_lr_scale=config.min_lr_scale)

    if name == "warmup_cosine":
        return lambda step, total: warmup_cosine_schedule(
            step, total, warmup_steps=config.warmup_steps, min_lr_scale=config.min_lr_scale,
        )

    if name == "step_decay":
        return lambda step, total: step_decay_schedule(
            step, total, step_size=config.step_size, gamma=config.gamma, min_lr_scale=config.min_lr_scale,
        )

    raise ValueError(f"Unknown scheduler name: {name}")
