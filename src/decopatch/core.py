from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import fmean
from typing import Iterable, Sequence

Number = int | float
Signal = Sequence[Number] | Sequence[Sequence[Number]]


@dataclass(frozen=True)
class Patch:
    values: tuple[float, ...]
    index: int
    level: int = 0

    @property
    def energy(self) -> float:
        return sum(value * value for value in self.values)

    @property
    def mean(self) -> float:
        return fmean(self.values) if self.values else 0.0

    @property
    def norm(self) -> float:
        return sqrt(self.energy)

    def centered(self) -> "Patch":
        mean = self.mean
        return Patch(tuple(value - mean for value in self.values), self.index, self.level)

    def scaled(self, factor: float) -> "Patch":
        return Patch(tuple(value * factor for value in self.values), self.index, self.level)


@dataclass(frozen=True)
class PatchGrid:
    patches: tuple[Patch, ...]
    patch_size: int
    stride: int
    length: int

    def __len__(self) -> int:
        return len(self.patches)

    def __iter__(self) -> Iterable[Patch]:
        return iter(self.patches)

    @property
    def coverage(self) -> float:
        if self.length <= 0:
            return 0.0
        seen = set()
        for patch in self.patches:
            start = patch.index * self.stride
            for offset in range(len(patch.values)):
                pos = start + offset
                if 0 <= pos < self.length:
                    seen.add(pos)
        return len(seen) / self.length

    def reconstruct(self) -> tuple[float, ...]:
        if self.length <= 0:
            return ()
        totals = [0.0 for _ in range(self.length)]
        counts = [0 for _ in range(self.length)]
        for patch in self.patches:
            start = patch.index * self.stride
            for offset, value in enumerate(patch.values):
                pos = start + offset
                if 0 <= pos < self.length:
                    totals[pos] += value
                    counts[pos] += 1
        return tuple(totals[idx] / counts[idx] if counts[idx] else 0.0 for idx in range(self.length))


@dataclass(frozen=True)
class Decomposition:
    base: tuple[float, ...]
    residual: tuple[float, ...]
    gates: tuple[float, ...]

    def compose(self) -> tuple[float, ...]:
        return tuple(base + gate * residual for base, residual, gate in zip(self.base, self.residual, self.gates))


def flatten_signal(signal: Signal) -> tuple[float, ...]:
    if not signal:
        return ()
    first = signal[0]
    if isinstance(first, (int, float)):
        return tuple(float(value) for value in signal)
    values: list[float] = []
    for row in signal:
        values.extend(float(value) for value in row)
    return tuple(values)


def normalize(values: Sequence[Number], eps: float = 1e-12) -> tuple[float, ...]:
    data = tuple(float(value) for value in values)
    if not data:
        return ()
    mean = fmean(data)
    variance = fmean((value - mean) * (value - mean) for value in data)
    scale = sqrt(variance + eps)
    return tuple((value - mean) / scale for value in data)


def chunk(values: Sequence[Number], size: int, stride: int | None = None, pad: bool = True) -> tuple[Patch, ...]:
    if size <= 0:
        raise ValueError("size must be positive")
    step = size if stride is None else stride
    if step <= 0:
        raise ValueError("stride must be positive")
    data = tuple(float(value) for value in values)
    if not data:
        return ()
    patches: list[Patch] = []
    index = 0
    pos = 0
    while pos < len(data):
        window = data[pos : pos + size]
        if len(window) < size:
            if not pad:
                break
            window = window + tuple(0.0 for _ in range(size - len(window)))
        patches.append(Patch(tuple(window), index))
        index += 1
        pos += step
    return tuple(patches)


def multiscale(values: Sequence[Number], sizes: Sequence[int], stride_ratio: float = 0.5) -> tuple[PatchGrid, ...]:
    data = tuple(float(value) for value in values)
    grids: list[PatchGrid] = []
    for level, size in enumerate(sizes):
        stride = max(1, int(size * stride_ratio))
        patches = tuple(Patch(patch.values, patch.index, level) for patch in chunk(data, size, stride))
        grids.append(PatchGrid(patches, size, stride, len(data)))
    return tuple(grids)


def decompose_patch(patch: Patch, temperature: float = 1.0) -> Decomposition:
    scale = max(temperature, 1e-6)
    mean = patch.mean
    base = tuple(mean for _ in patch.values)
    residual = tuple(value - mean for value in patch.values)
    gates = tuple(1.0 / (1.0 + abs(value) / scale) for value in residual)
    return Decomposition(base, residual, gates)
