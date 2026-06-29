from __future__ import annotations

from decopatch.models.configs import ModelConfig

_REGISTRY: dict[str, ModelConfig] = {}


def register_model(config: ModelConfig) -> None:
    if config.name in _REGISTRY:
        raise ValueError(f"model already registered: {config.name}")
    _REGISTRY[config.name] = config


def get_model_config(name: str) -> ModelConfig:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(available_models())
        raise KeyError(f"unknown model {name}; available: {choices}") from exc


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
