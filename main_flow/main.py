from __future__ import annotations

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
    runtime = load_runtime_config()
    print(f"Mode: {runtime.mode}")
    print(f"Benchmarks: {', '.join(runtime.benchmarks)}")
    run_mode(runtime)


if __name__ == "__main__":
    main()
