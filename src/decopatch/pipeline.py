from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decopatch.data import Dataset, make_wave_dataset
from decopatch.evaluation import EvaluationResult, Evaluator
from decopatch.io import load_dataset, write_json
from decopatch.models.factory import build_model


@dataclass(frozen=True)
class PipelineConfig:
    model: str = "decopatch_tiny"
    dataset: str | None = None
    output: str | None = None
    standardize: bool = False


class DeCoPatchPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = build_model(config.model)
        self.evaluator = Evaluator()

    def load(self) -> Dataset:
        if self.config.dataset:
            dataset = load_dataset(self.config.dataset)
        else:
            dataset = make_wave_dataset()
        return dataset.map_signals(self.config.standardize)

    def run(self) -> EvaluationResult:
        dataset = self.load()
        result = self.evaluator.evaluate(self.model, dataset)
        if self.config.output:
            self.save(result, self.config.output)
        return result

    def save(self, result: EvaluationResult, output: str | Path) -> None:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "metrics": result.metrics,
            "predictions": [list(value) if isinstance(value, tuple) else value for value in result.predictions],
            "targets": [list(value) if isinstance(value, tuple) else value for value in result.targets],
        }
        write_json(output, payload)
