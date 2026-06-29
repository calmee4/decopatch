from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    patch_sizes: tuple[int, ...] = (8,)
    hidden_dim: int = 16
    depth: int = 2
    heads: int = 2
    output_dim: int = 1
    temperature: float = 1.0
    residual_weight: float = 0.5
    fusion: str = "mean"
    head: str = "regression"
