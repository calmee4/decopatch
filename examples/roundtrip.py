from decopatch.pipeline import DeCoPatchPipeline, PipelineConfig


def main() -> int:
    result = DeCoPatchPipeline(PipelineConfig(dataset="examples/sample_matrix.json")).run()
    for key, value in result.metrics.items():
        print(f"{key}: {value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
