from __future__ import annotations

from decopatch.models.configs import ModelConfig
from decopatch.models.registry import register_model


def register_default_models() -> None:
    defaults = (
        ModelConfig("decopatch_tiny", patch_sizes=(6,), hidden_dim=12, depth=1, heads=1, fusion="mean"),
        ModelConfig("decopatch_small", patch_sizes=(6, 12), hidden_dim=18, depth=2, heads=2, fusion="meanmax"),
        ModelConfig("decopatch_base", patch_sizes=(8, 16, 24), hidden_dim=24, depth=3, heads=3, fusion="meanmax"),
        ModelConfig("decopatch_multi", patch_sizes=(4, 8, 16), hidden_dim=20, depth=2, heads=2, fusion="max"),
        ModelConfig("decopatch_cls", patch_sizes=(8, 16), hidden_dim=18, depth=2, heads=2, output_dim=2, head="classification"),
        ModelConfig("decopatch_rec32", patch_sizes=(8,), hidden_dim=20, depth=2, heads=2, output_dim=32, head="reconstruction"),
    )
    for config in defaults:
        try:
            register_model(config)
        except ValueError:
            pass


register_default_models()
