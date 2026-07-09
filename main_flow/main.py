from __future__ import annotations

import argparse

from configs.runtime import load_runtime_config
from .context import build_context
from .phase import run_phase
from .validation import run_train_with_validation


def run_mode(runtime):
    ctx = build_context(runtime)

    for benchmark in runtime.benchmarks:
        if runtime.mode in {"train_val", "train_val_test"}:
            run_train_with_validation(benchmark, ctx)
        elif runtime.mode in {"train", "train_test"}:
            run_phase("train", benchmark, ctx)

        if runtime.mode == "val":
            run_phase("val", benchmark, ctx)

        if runtime.mode in {"test", "train_test", "train_val_test"}:
            run_phase("test", benchmark, ctx)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MMAgent benchmark pipeline.")
    parser.add_argument("--config", default=None, help="Runtime YAML path. Defaults to configs/runtime.yaml.")
    args = parser.parse_args()

    runtime = load_runtime_config(args.config)
    print(f"Mode: {runtime.mode}")
    print(f"Benchmarks: {', '.join(runtime.benchmarks)}")
    run_mode(runtime)


if __name__ == "__main__":
    main()
