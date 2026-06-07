"""Command-line entry for benchmark train/validation/test flow."""

from __future__ import annotations

from configs.runtime import load_runtime_config
from .context import build_context
from .phase import run_phase
from .validation import run_train_with_validation
from utils.env import env_bool


def _run_mode(mode: str, benchmarks: list[str]) -> None:
    ctx = build_context(len(benchmarks))
    for benchmark in benchmarks:
        if env_bool("ENABLE_VALIDATION", False) and mode in {"train", "train_test", "train_val_test"}:
            run_train_with_validation(benchmark, ctx)
        elif mode in {"train", "train_test", "train_val_test"}:
            run_phase("train", benchmark, ctx)

        if mode == "val":
            run_phase("val", benchmark, ctx)

        if mode in {"test", "train_test", "train_val_test"}:
            run_phase("test", benchmark, ctx)


def main() -> None:
    runtime = load_runtime_config()
    print(f"Runtime config: {runtime.path}")
    print(f"Mode: {runtime.mode}")
    print(f"Benchmarks: {', '.join(runtime.benchmarks)}")
    _run_mode(runtime.mode, runtime.benchmarks)


if __name__ == "__main__":
    main()
