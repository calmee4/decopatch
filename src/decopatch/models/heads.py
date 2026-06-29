from __future__ import annotations

from math import exp
from typing import Sequence

from decopatch.models.layers import Linear, Vector


class RegressionHead:
    def __init__(self, in_dim: int, out_dim: int = 1, seed: int = 0):
        self.linear = Linear(in_dim, out_dim, seed)

    def __call__(self, vector: Sequence[float]) -> float | Vector:
        output = self.linear(vector)
        return output[0] if len(output) == 1 else output


class ClassificationHead:
    def __init__(self, in_dim: int, classes: int = 2, seed: int = 0):
        self.linear = Linear(in_dim, classes, seed)

    def __call__(self, vector: Sequence[float]) -> Vector:
        logits = self.linear(vector)
        center = max(logits)
        scores = tuple(exp(value - center) for value in logits)
        total = sum(scores)
        return tuple(value / total for value in scores)


class ReconstructionHead:
    def __init__(self, in_dim: int, length: int, seed: int = 0):
        self.linear = Linear(in_dim, length, seed)

    def __call__(self, vector: Sequence[float]) -> Vector:
        return self.linear(vector)


def build_head(name: str, in_dim: int, output_dim: int, seed: int = 0):
    if name == "classification":
        return ClassificationHead(in_dim, output_dim, seed)
    if name == "reconstruction":
        return ReconstructionHead(in_dim, output_dim, seed)
    return RegressionHead(in_dim, output_dim, seed)
