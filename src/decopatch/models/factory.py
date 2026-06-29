from __future__ import annotations

from decopatch.models.configs import ModelConfig
from decopatch.models.registry import get_model_config
from decopatch.models.variants import DeCoPatchClassifier, DeCoPatchModel, DeCoPatchReconstructor, DeCoPatchRegressor
from decopatch.models import zoo as _zoo


def build_model(name: str | ModelConfig) -> DeCoPatchModel:
    config = name if isinstance(name, ModelConfig) else get_model_config(name)
    if config.head == "classification":
        return DeCoPatchClassifier(config)
    if config.head == "reconstruction":
        return DeCoPatchReconstructor(config)
    return DeCoPatchRegressor(config)
