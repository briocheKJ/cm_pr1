"""
Global mode switch: "student" or "teacher".

- student mode: uses student-implemented files (stubs by default)
- teacher mode: loads reference solutions from _teacher_solutions/
"""
from __future__ import annotations

_current_mode: str = "student"


def set_mode(mode: str) -> None:
    global _current_mode
    if mode not in ("student", "teacher"):
        raise ValueError(f"Mode must be 'student' or 'teacher', got '{mode}'")
    _current_mode = mode


def get_mode() -> str:
    return _current_mode


def is_teacher() -> bool:
    return _current_mode == "teacher"
