from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from decopatch.evaluation import Evaluator
from decopatch.io import write_json
from decopatch.models.factory import build_model
from decopatch.data import Dataset, make_wave_dataset


@dataclass(frozen=True)
class ExperimentConfig:
    models: tuple[str, ...] = ("decopatch_tiny", "decopatch_small", "decopatch_multi")
    seeds: tuple[int, ...] = (3, 7, 13)
    length: int = 48
    count: int = 32


@dataclass(frozen=True)
class ExperimentRun:
    model: str
    seed: int
    metrics: dict[str, float]


def run_experiment(config: ExperimentConfig = ExperimentConfig()) -> tuple[ExperimentRun, ...]:
    runs = []
    evaluator = Evaluator()
    for seed in config.seeds:
        dataset = make_wave_dataset(config.count, config.length, seed)
        for model_name in config.models:
            model = build_model(model_name)
            result = evaluator.evaluate(model, dataset)
            runs.append(ExperimentRun(model_name, seed, result.metrics))
    return tuple(runs)


def aggregate(runs: Sequence[ExperimentRun]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[ExperimentRun]] = {}
    for run in runs:
        grouped.setdefault(run.model, []).append(run)
    summary: dict[str, dict[str, float]] = {}
    for model, items in grouped.items():
        keys = sorted({key for item in items for key in item.metrics})
        summary[model] = {
            key: sum(item.metrics[key] for item in items if key in item.metrics) / len(items)
            for key in keys
        }
    return summary


def save_experiment(path: str | Path, runs: Sequence[ExperimentRun]) -> None:
    write_json(
        path,
        {
            "runs": [{"model": run.model, "seed": run.seed, "metrics": run.metrics} for run in runs],
            "summary": aggregate(runs),
        },
    )


def benchmark_dataset() -> Dataset:
    return make_wave_dataset(count=64, length=64, seed=11)
