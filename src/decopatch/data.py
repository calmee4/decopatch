from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterator, Sequence

from decopatch.core import flatten_signal, normalize


@dataclass(frozen=True)
class Sample:
    signal: tuple[float, ...]
    target: float | tuple[float, ...]
    name: str = ""


class Dataset:
    def __init__(self, samples: Sequence[Sample]):
        self.samples = tuple(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]

    def map_signals(self, standardize: bool = False) -> "Dataset":
        mapped = []
        for sample in self.samples:
            signal = normalize(sample.signal) if standardize else tuple(float(value) for value in sample.signal)
            mapped.append(Sample(signal, sample.target, sample.name))
        return Dataset(mapped)

    def split(self, ratio: float = 0.8) -> tuple["Dataset", "Dataset"]:
        if not 0.0 < ratio < 1.0:
            raise ValueError("ratio must be between zero and one")
        cut = int(len(self.samples) * ratio)
        return Dataset(self.samples[:cut]), Dataset(self.samples[cut:])

    def batch(self, size: int) -> Iterator[tuple[Sample, ...]]:
        if size <= 0:
            raise ValueError("size must be positive")
        for pos in range(0, len(self.samples), size):
            yield self.samples[pos : pos + size]


def make_wave_dataset(count: int = 32, length: int = 48, seed: int = 7) -> Dataset:
    rng = Random(seed)
    samples = []
    for idx in range(count):
        phase = rng.random()
        frequency = 1 + idx % 5
        signal = tuple(((pos / max(length - 1, 1)) * frequency + phase) % 1.0 for pos in range(length))
        target = sum(signal) / len(signal)
        samples.append(Sample(signal, target, f"wave_{idx:04d}"))
    return Dataset(samples)


def samples_from_records(records: Sequence[dict]) -> Dataset:
    samples = []
    for idx, record in enumerate(records):
        signal = flatten_signal(record.get("signal", ()))
        target = record.get("target", 0.0)
        name = str(record.get("name", f"sample_{idx:04d}"))
        if isinstance(target, list):
            parsed_target: float | tuple[float, ...] = tuple(float(value) for value in target)
        else:
            parsed_target = float(target)
        samples.append(Sample(signal, parsed_target, name))
    return Dataset(samples)
