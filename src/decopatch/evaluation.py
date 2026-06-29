from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from decopatch.data import Dataset
from decopatch.metrics import ScalarOrVector, summarize


class Predictor(Protocol):
    def predict(self, signal: Sequence[float]) -> ScalarOrVector:
        ...


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    predictions: tuple[ScalarOrVector, ...]
    targets: tuple[ScalarOrVector, ...]


class Evaluator:
    def __init__(self, metrics: Sequence[str] | None = None):
        self.metrics = tuple(metrics or ("mae", "mse", "rmse", "accuracy"))

    def evaluate(self, model: Predictor, dataset: Dataset) -> EvaluationResult:
        predictions = tuple(model.predict(sample.signal) for sample in dataset)
        targets = tuple(sample.target for sample in dataset)
        all_metrics = summarize(predictions, targets)
        selected = {name: all_metrics[name] for name in self.metrics if name in all_metrics}
        return EvaluationResult(selected, predictions, targets)
