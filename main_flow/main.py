"""Command-line entry for benchmark train/validation/test flow."""

from __future__ import annotations

from configs.runtime import RUNTIME_CONFIG_PATH, load_runtime_config
from .context import build_context
from .phase import run_phase
from .validation import run_train_with_validation


def _run_mode(runtime) -> None:
    mode = runtime.mode
    benchmarks = runtime.benchmarks
    ctx = build_context(len(benchmarks), runtime)
    for benchmark in benchmarks:
        if mode in {"train_val", "train_val_test"}:
            run_train_with_validation(benchmark, ctx)
        elif mode in {"train", "train_test"}:
            run_phase("train", benchmark, ctx)

        if mode == "val":
            run_phase("val", benchmark, ctx)

        if mode in {"test", "train_test", "train_val_test"}:
            run_phase("test", benchmark, ctx)


def main() -> None:
    runtime = load_runtime_config()
    print(f"Runtime config: {RUNTIME_CONFIG_PATH}")
    print(f"Mode: {runtime.mode}")
    print(f"Benchmarks: {', '.join(runtime.benchmarks)}")
    _run_mode(runtime)


if __name__ == "__main__":
    main()
