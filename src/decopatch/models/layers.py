from __future__ import annotations

from math import exp, sqrt
from random import Random
from statistics import fmean
from typing import Sequence

Vector = tuple[float, ...]
Matrix = tuple[Vector, ...]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def add(left: Sequence[float], right: Sequence[float]) -> Vector:
    return tuple(a + b for a, b in zip(left, right))


def scale(vector: Sequence[float], factor: float) -> Vector:
    return tuple(value * factor for value in vector)


def mean_pool(vectors: Sequence[Sequence[float]]) -> Vector:
    rows = tuple(tuple(row) for row in vectors)
    if not rows:
        return ()
    width = len(rows[0])
    return tuple(fmean(row[idx] for row in rows) for idx in range(width))


def max_pool(vectors: Sequence[Sequence[float]]) -> Vector:
    rows = tuple(tuple(row) for row in vectors)
    if not rows:
        return ()
    width = len(rows[0])
    return tuple(max(row[idx] for row in rows) for idx in range(width))


def softmax(values: Sequence[float]) -> Vector:
    if not values:
        return ()
    center = max(values)
    weights = tuple(exp(value - center) for value in values)
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def layer_norm(vector: Sequence[float], eps: float = 1e-6) -> Vector:
    data = tuple(vector)
    if not data:
        return ()
    mean = fmean(data)
    var = fmean((value - mean) * (value - mean) for value in data)
    denom = sqrt(var + eps)
    return tuple((value - mean) / denom for value in data)


def gelu(vector: Sequence[float]) -> Vector:
    return tuple(0.5 * value * (1.0 + value / sqrt(1.0 + value * value)) for value in vector)


class Linear:
    def __init__(self, in_dim: int, out_dim: int, seed: int = 0):
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("dimensions must be positive")
        rng = Random(seed + in_dim * 131 + out_dim * 17)
        limit = 1.0 / sqrt(in_dim)
        self.weight: Matrix = tuple(
            tuple(rng.uniform(-limit, limit) for _ in range(in_dim)) for _ in range(out_dim)
        )
        self.bias: Vector = tuple(rng.uniform(-limit, limit) for _ in range(out_dim))

    def __call__(self, vector: Sequence[float]) -> Vector:
        values = tuple(vector)
        return tuple(dot(row, values) + bias for row, bias in zip(self.weight, self.bias))


class ResidualBlock:
    def __init__(self, dim: int, seed: int = 0):
        self.proj_in = Linear(dim, dim, seed)
        self.proj_out = Linear(dim, dim, seed + 1)

    def __call__(self, vector: Sequence[float]) -> Vector:
        hidden = gelu(self.proj_in(layer_norm(vector)))
        return add(vector, self.proj_out(hidden))


class PatchAttention:
    def __init__(self, dim: int, heads: int = 2, seed: int = 0):
        if heads <= 0:
            raise ValueError("heads must be positive")
        self.dim = dim
        self.heads = heads
        self.query = Linear(dim, dim, seed)
        self.key = Linear(dim, dim, seed + 1)
        self.value = Linear(dim, dim, seed + 2)
        self.out = Linear(dim, dim, seed + 3)

    def __call__(self, vectors: Sequence[Sequence[float]]) -> tuple[Vector, ...]:
        rows = tuple(tuple(row) for row in vectors)
        if not rows:
            return ()
        queries = tuple(self.query(row) for row in rows)
        keys = tuple(self.key(row) for row in rows)
        values = tuple(self.value(row) for row in rows)
        outputs = []
        scale_factor = sqrt(max(self.dim, 1))
        for query in queries:
            weights = softmax(tuple(dot(query, key) / scale_factor for key in keys))
            merged = tuple(sum(weight * value[idx] for weight, value in zip(weights, values)) for idx in range(self.dim))
            outputs.append(self.out(merged))
        return tuple(outputs)
