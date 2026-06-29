from __future__ import annotations

from typing import Sequence

from decopatch.core import decompose_patch, multiscale
from decopatch.models.configs import ModelConfig
from decopatch.models.layers import Linear, PatchAttention, ResidualBlock, Vector, add, layer_norm, max_pool, mean_pool


class PatchEmbedder:
    def __init__(self, patch_size: int, hidden_dim: int, seed: int = 0):
        self.patch_size = patch_size
        self.project = Linear(patch_size * 3, hidden_dim, seed)

    def __call__(self, values: Sequence[float], temperature: float) -> Vector:
        from decopatch.core import Patch

        patch = Patch(tuple(values), 0)
        decomposition = decompose_patch(patch, temperature)
        features = tuple(decomposition.base + decomposition.residual + decomposition.gates)
        return self.project(features)


class DecompositionEncoder:
    def __init__(self, config: ModelConfig, seed: int = 0):
        self.config = config
        self.embedders = {
            size: PatchEmbedder(size, config.hidden_dim, seed + idx * 19)
            for idx, size in enumerate(config.patch_sizes)
        }
        self.attention = PatchAttention(config.hidden_dim, config.heads, seed + 101)
        self.blocks = tuple(ResidualBlock(config.hidden_dim, seed + 211 + idx) for idx in range(config.depth))

    def encode_patches(self, signal: Sequence[float]) -> tuple[Vector, ...]:
        vectors = []
        for grid in multiscale(signal, self.config.patch_sizes):
            embedder = self.embedders[grid.patch_size]
            for patch in grid:
                vectors.append(embedder(patch.values, self.config.temperature))
        if not vectors:
            return ()
        attended = self.attention(vectors)
        encoded = []
        for vector in attended:
            current = vector
            for block in self.blocks:
                current = block(current)
            encoded.append(layer_norm(current))
        return tuple(encoded)

    def pool(self, vectors: Sequence[Sequence[float]]) -> Vector:
        if self.config.fusion == "max":
            return max_pool(vectors)
        if self.config.fusion == "meanmax":
            return add(mean_pool(vectors), max_pool(vectors))
        return mean_pool(vectors)

    def __call__(self, signal: Sequence[float]) -> Vector:
        vectors = self.encode_patches(signal)
        if not vectors:
            return tuple(0.0 for _ in range(self.config.hidden_dim))
        return self.pool(vectors)
