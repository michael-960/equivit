from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Callable
import textwrap


class DataTransform:
    """
    A class representing a sequence of data transformations, each with an
    associated role (either "augmentation" or "preprocessing").

    The roles are usually labeled manually.

    This is to record whether there is data augmentation.
    """
    VALID_ROLES = {"augmentation", "preprocessing"}

    def __init__(self, steps: Iterable[Mapping[str, Any]]):
        self.steps = []

        for i, step in enumerate(steps):
            role = step["role"]
            transform = step["transform"]

            if role not in self.VALID_ROLES:
                raise ValueError(
                    f"Invalid role at transform step {i}: {role!r}. "
                    f"Expected one of {sorted(self.VALID_ROLES)}."
                )

            if not callable(transform):
                raise TypeError(
                    f"Transform at step {i} is not callable: {transform!r}"
                )

            self.steps.append({
                "role": role,
                "transform": transform,
            })

        self.has_augmentation = any(
            step["role"] == "augmentation"
            for step in self.steps
        )

        self.num_augmentation_steps = sum(
            step["role"] == "augmentation"
            for step in self.steps
        )

        self.num_preprocessing_steps = sum(
            step["role"] == "preprocessing"
            for step in self.steps
        )

    def __call__(self, x: Any) -> Any:
        for step in self.steps:
            x = step["transform"](x)
        return x
    
    @staticmethod
    def _safe_transform_repr(transform: Any, *, max_chars: int = 120) -> str:
        try:
            text = repr(transform)
        except Exception as exc:
            text = f"<repr failed: {type(exc).__name__}: {exc}>"

        # Make multiline reprs compact.
        text = " ".join(text.split())

        # Shorten very long transform reprs.
        return textwrap.shorten(
            text,
            width=max_chars,
            placeholder="...",
        )

    def __repr__(self) -> str:
        max_steps = 12
        max_transform_chars = 120

        shown_steps = self.steps[:max_steps]

        step_reprs = []
        for i, step in enumerate(shown_steps):
            role = step.get("role", "<missing>")
            transform = step.get("transform", "<missing>")

            transform_cls = type(transform).__name__
            transform_repr = self._safe_transform_repr(
                transform,
                max_chars=max_transform_chars,
            )

            step_reprs.append(
                f"{i}: {role} -> {transform_cls}({transform_repr})"
            )

        if len(self.steps) > max_steps:
            step_reprs.append(f"... {len(self.steps) - max_steps} more step(s)")

        steps_text = ", ".join(step_reprs)

        return (
            f"{type(self).__name__}("
            f"num_steps={len(self.steps)}, "
            f"has_augmentation={self.has_augmentation}, "
            f"num_augmentation_steps={self.num_augmentation_steps}, "
            f"num_preprocessing_steps={self.num_preprocessing_steps}, "
            f"steps=[{steps_text}]"
            f")"
        )
    