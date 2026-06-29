from __future__ import annotations

from math import sqrt
from statistics import fmean
from typing import Sequence

ScalarOrVector = float | Sequence[float]


def as_vector(value: ScalarOrVector) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        return (float(value),)
    return tuple(float(item) for item in value)


def mae(predictions: Sequence[ScalarOrVector], targets: Sequence[ScalarOrVector]) -> float:
    values = []
    for pred, target in zip(predictions, targets):
        pred_vec = as_vector(pred)
        target_vec = as_vector(target)
        values.extend(abs(a - b) for a, b in zip(pred_vec, target_vec))
    return fmean(values) if values else 0.0


def mse(predictions: Sequence[ScalarOrVector], targets: Sequence[ScalarOrVector]) -> float:
    values = []
    for pred, target in zip(predictions, targets):
        pred_vec = as_vector(pred)
        target_vec = as_vector(target)
        values.extend((a - b) * (a - b) for a, b in zip(pred_vec, target_vec))
    return fmean(values) if values else 0.0


def rmse(predictions: Sequence[ScalarOrVector], targets: Sequence[ScalarOrVector]) -> float:
    return sqrt(mse(predictions, targets))


def accuracy(predictions: Sequence[ScalarOrVector], targets: Sequence[ScalarOrVector], tolerance: float = 0.5) -> float:
    total = 0
    correct = 0
    for pred, target in zip(predictions, targets):
        pred_vec = as_vector(pred)
        target_vec = as_vector(target)
        if not pred_vec or not target_vec:
            continue
        total += 1
        if abs(pred_vec[0] - target_vec[0]) <= tolerance:
            correct += 1
    return correct / total if total else 0.0


def summarize(predictions: Sequence[ScalarOrVector], targets: Sequence[ScalarOrVector]) -> dict[str, float]:
    return {
        "mae": mae(predictions, targets),
        "mse": mse(predictions, targets),
        "rmse": rmse(predictions, targets),
        "accuracy": accuracy(predictions, targets),
    }
