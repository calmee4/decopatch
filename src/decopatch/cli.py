from __future__ import annotations

import argparse
from typing import Sequence

from decopatch.experiments import ExperimentConfig, run_experiment, save_experiment
from decopatch.models.registry import available_models
from decopatch.pipeline import DeCoPatchPipeline, PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="decopatch")
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model", default="decopatch_tiny")
    run_parser.add_argument("--dataset")
    run_parser.add_argument("--output")
    run_parser.add_argument("--standardize", action="store_true")
    list_parser = subparsers.add_parser("models")
    list_parser.set_defaults(models=True)
    exp_parser = subparsers.add_parser("experiment")
    exp_parser.add_argument("--output", default="outputs/experiment.json")
    exp_parser.add_argument("--models", nargs="*", default=["decopatch_tiny", "decopatch_small", "decopatch_multi"])
    exp_parser.add_argument("--seeds", nargs="*", type=int, default=[3, 7, 13])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "models":
        for name in available_models():
            print(name)
        return 0
    if args.command == "experiment":
        runs = run_experiment(ExperimentConfig(models=tuple(args.models), seeds=tuple(args.seeds)))
        save_experiment(args.output, runs)
        print(args.output)
        return 0
    config = PipelineConfig(
        model=getattr(args, "model", "decopatch_tiny"),
        dataset=getattr(args, "dataset", None),
        output=getattr(args, "output", None),
        standardize=getattr(args, "standardize", False),
    )
    result = DeCoPatchPipeline(config).run()
    for key, value in result.metrics.items():
        print(f"{key}: {value:.6f}")
    return 0
