from __future__ import annotations

from typing import Callable

from config import SchedulerConfig, is_teacher


def constant_schedule(step: int, total_steps: int) -> float:
    return 1.0


def cosine_schedule(step: int, total_steps: int, min_lr_scale: float = 0.01) -> float:
    raise NotImplementedError("TODO: implement cosine_schedule")


def warmup_cosine_schedule(step: int, total_steps: int, warmup_steps: int, min_lr_scale: float = 0.01) -> float:
    raise NotImplementedError("TODO: implement warmup_cosine_schedule")


def step_decay_schedule(step: int, total_steps: int, step_size: int = 100, gamma: float = 0.5, min_lr_scale: float = 0.01) -> float:
    raise NotImplementedError("TODO: implement step_decay_schedule")


def build_scheduler(config: SchedulerConfig) -> Callable[[int, int], float]:
    name = config.name

    if name == "constant":
        return constant_schedule

    if is_teacher():
        from _teacher_solutions.schedulers import build_teacher_scheduler
        return build_teacher_scheduler(config)

    if name == "cosine":
        return lambda step, total: cosine_schedule(step, total, min_lr_scale=0.01)
    if name == "warmup_cosine":
        return lambda step, total: warmup_cosine_schedule(
            step, total, warmup_steps=0, min_lr_scale=0.01,
        )
    if name == "step_decay":
        return lambda step, total: step_decay_schedule(
            step, total, step_size=100, gamma=0.5, min_lr_scale=0.01,
        )

    raise ValueError(f"Unknown scheduler name: {name}")
