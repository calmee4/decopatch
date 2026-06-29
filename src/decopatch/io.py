from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from decopatch.data import Dataset, Sample, samples_from_records


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_dataset(path: str | Path) -> Dataset:
    payload = read_json(path)
    records = payload["samples"] if isinstance(payload, dict) and "samples" in payload else payload
    if not isinstance(records, list):
        raise ValueError("dataset must be a list or contain a samples list")
    return samples_from_records(records)


def save_dataset(path: str | Path, dataset: Dataset) -> None:
    records = []
    for sample in dataset:
        target = list(sample.target) if isinstance(sample.target, tuple) else sample.target
        records.append({"name": sample.name, "signal": list(sample.signal), "target": target})
    write_json(path, {"samples": records})


def sample_to_json(sample: Sample) -> dict[str, Any]:
    target = list(sample.target) if isinstance(sample.target, tuple) else sample.target
    return {"name": sample.name, "signal": list(sample.signal), "target": target}
