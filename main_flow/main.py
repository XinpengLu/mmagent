from __future__ import annotations

import argparse

from configs.runtime import load_runtime_config
from .context import (
    build_context,
    full_hive_test_memory_map,
    full_hive_test_memory_root,
    full_hive_test_memory_run,
    reuse_train_map,
    validate_reuse_train_order,
)
from .phase import run_phase
from .validation import run_train_with_validation


def run_mode(runtime):
    ctx = build_context(runtime)
    reuse_map = reuse_train_map(runtime.settings)
    full_memory_map = full_hive_test_memory_map(runtime.settings)
    full_memory_run = full_hive_test_memory_run(runtime.settings)

    for benchmark in runtime.benchmarks:
        reuse_source = reuse_map.get(benchmark, "")
        if reuse_source and runtime.mode in {"train", "train_test", "train_val", "train_val_test"}:
            print(f"[Reuse] Skipping {benchmark} train/val; test will reuse {reuse_source} memory.")
        elif runtime.mode in {"train_val", "train_val_test"}:
            run_train_with_validation(benchmark, ctx)
        elif runtime.mode in {"train", "train_test"}:
            run_phase("train", benchmark, ctx)

        if runtime.mode == "val":
            run_phase("val", benchmark, ctx)

        if runtime.mode in {"test", "train_test", "train_val_test"}:
            phase_env = None
            full_memory_source = full_memory_map.get(benchmark, "")
            if full_memory_source:
                source_memory = full_hive_test_memory_root(ctx, full_memory_run, full_memory_source)
                phase_env = {
                    "MEMORY_ROOT_OVERRIDE": str(source_memory),
                    "MEMORY_NAME_OVERRIDE": full_memory_source,
                    "REUSE_TRAIN_BENCHMARK": f"{full_memory_run}/{full_memory_source}",
                    "REQUIRE_NONEMPTY_MEMORY": "true",
                }
                print(f"[Reuse] Testing {benchmark} with full HIVE memory: {source_memory}")
            elif reuse_source:
                phase_env = {
                    "MEMORY_ROOT_OVERRIDE": str(ctx.memory_root_base / reuse_source / "memory"),
                    "MEMORY_NAME_OVERRIDE": reuse_source,
                    "REUSE_TRAIN_BENCHMARK": reuse_source,
                }
                print(f"[Reuse] Testing {benchmark} with {reuse_source} memory.")
            run_phase("test", benchmark, ctx, phase_env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MMAgent benchmark pipeline.")
    parser.add_argument("--config", default=None, help="Runtime YAML path. Defaults to configs/runtime.yaml.")
    args = parser.parse_args()

    runtime = load_runtime_config(args.config)
    validate_reuse_train_order(runtime.benchmarks, runtime.settings)
    print(f"Mode: {runtime.mode}")
    print(f"Benchmarks: {', '.join(runtime.benchmarks)}")
    run_mode(runtime)


if __name__ == "__main__":
    main()
