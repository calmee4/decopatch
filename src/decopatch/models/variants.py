from __future__ import annotations

from typing import Sequence

from decopatch.metrics import ScalarOrVector
from decopatch.models.backbone import DecompositionEncoder
from decopatch.models.configs import ModelConfig
from decopatch.models.heads import build_head
from decopatch.models.layers import Vector


class DeCoPatchModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.encoder = DecompositionEncoder(config)
        self.head = build_head(config.head, config.hidden_dim, config.output_dim, 409)

    def encode(self, signal: Sequence[float]) -> Vector:
        return self.encoder(signal)

    def predict(self, signal: Sequence[float]) -> ScalarOrVector:
        return self.head(self.encode(signal))

    def __call__(self, signal: Sequence[float]) -> ScalarOrVector:
        return self.predict(signal)


class DeCoPatchRegressor(DeCoPatchModel):
    pass


class DeCoPatchClassifier(DeCoPatchModel):
    pass


class DeCoPatchReconstructor(DeCoPatchModel):
    pass


class EnsembleDeCoPatch:
    def __init__(self, configs: Sequence[ModelConfig]):
        if not configs:
            raise ValueError("configs must not be empty")
        self.models = tuple(DeCoPatchModel(config) for config in configs)

    def predict(self, signal: Sequence[float]) -> ScalarOrVector:
        outputs = tuple(model.predict(signal) for model in self.models)
        first = outputs[0]
        if isinstance(first, tuple):
            width = len(first)
            return tuple(sum(output[idx] for output in outputs if isinstance(output, tuple)) / len(outputs) for idx in range(width))
        return sum(float(output) for output in outputs) / len(outputs)
