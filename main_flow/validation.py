"""Staged train-validation workflow."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .context import MainFlowContext, _count_json_items
from .phase import run_phase
from .validation_memory import apply_validation_feedback, calibrate_validation_status, select_best
from .validation_reflection import reflect_validation_failures
from validation_tools.history import record_validation_stage
from validation_tools.reporting import write_validation_report
from utils.artifacts import DATASET_SUMMARY_FILENAME, copy_file_contents, copy_tree_contents


def _validation_setting(ctx: MainFlowContext, key: str, default):
    section = ctx.settings.get("validation", {}) if isinstance(ctx.settings, dict) else {}
    return section.get(key, default) if isinstance(section, dict) else default


def _ctx_setting(ctx: MainFlowContext, path: str, default):
    current: Any = ctx.settings if isinstance(ctx.settings, dict) else {}
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _rollout_setting(ctx: MainFlowContext, key: str, default):
    section = ctx.settings.get("rollout", {}) if isinstance(ctx.settings, dict) else {}
    return section.get(key, default) if isinstance(section, dict) else default


@dataclass(frozen=True)
class _ValidationPaths:
    train_memory_root: Path
    stage_dir: Path
    benchmark_output_root: Path
    train_output_root: Path
    train_stage_root: Path
    val_output_root: Path
    analysis_output_root: Path
    snapshot_root: Path
    history_path: Path
    selected_json: Path
    val_file: Path


def _copy_memory_snapshot(source_root: Path, target_root: Path) -> None:
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.parent.mkdir(parents=True, exist_ok=True)
    if source_root.exists():
        copy_tree_contents(source_root, target_root)
    else:
        target_root.mkdir(parents=True, exist_ok=True)


def _load_stage_manifest(stage_dir: Path) -> list[dict[str, Any]]:
    manifest_path = stage_dir / "stages.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"Error: missing fixed stage manifest: {manifest_path}\n"
            "Run benchmark/build_benchmark_subset.py before training."
        )
    with manifest_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise SystemExit(f"Error: {manifest_path} must contain a JSON list.")
    for row in rows:
        path = Path(str(row.get("path", "")))
        if not path.is_absolute():
            path = stage_dir / path
        row["resolved_path"] = str(path)
    return rows


def _reflection_has_writeback(path: Path) -> bool:
    if not path.exists():
        return False
    row = json.loads(path.read_text(encoding="utf-8"))
    return bool(row.get("write_back")) and int(row.get("operations", 0) or 0) > 0


def _validation_paths(benchmark: str, ctx: MainFlowContext) -> _ValidationPaths:
    memory_name = os.environ.get("MEMORY_NAME") or benchmark
    memory_root_base = os.environ.get("MEMORY_ROOT_BASE", "")
    default_memory_root = (
        Path(memory_root_base) / memory_name / "memory"
        if memory_root_base
        else ctx.output_root / benchmark / "memory"
    )
    train_memory_root = Path(os.environ.get("MEMORY_ROOT") or default_memory_root)

    stage_dir = ctx.split_root / benchmark / "stages"
    benchmark_output_root = ctx.output_root / benchmark / "output"
    train_output_root = benchmark_output_root / "train"
    train_stage_root = train_output_root / "stages"
    val_output_root = benchmark_output_root / "val"
    analysis_output_root = benchmark_output_root / "analysis"
    snapshot_root = Path(os.environ.get("VALIDATION_SNAPSHOT_ROOT") or ctx.output_root / benchmark / "snapshots")
    history_path = Path(os.environ.get("VALIDATION_HISTORY") or benchmark_output_root / "validation_history.jsonl")
    selected_json = Path(os.environ.get("VALIDATION_SELECTED_JSON") or benchmark_output_root / "best_validation_memory.json")
    val_file = ctx.split_root / benchmark / "val.json"
    return _ValidationPaths(
        train_memory_root=train_memory_root,
        stage_dir=stage_dir,
        benchmark_output_root=benchmark_output_root,
        train_output_root=train_output_root,
        train_stage_root=train_stage_root,
        val_output_root=val_output_root,
        analysis_output_root=analysis_output_root,
        snapshot_root=snapshot_root,
        history_path=history_path,
        selected_json=selected_json,
        val_file=val_file,
    )


def _validation_stages_to_run(paths: _ValidationPaths) -> tuple[list[dict[str, Any]], int]:
    if not paths.val_file.exists():
        raise SystemExit(
            f"Error: missing fixed validation split: {paths.val_file}\n"
            "Run benchmark/build_benchmark_subset.py before training."
        )
    stages = _load_stage_manifest(paths.stage_dir)
    resume_stage = 1
    if resume_stage < 1:
        raise SystemExit("Error: VALIDATION_RESUME_STAGE must be >= 1")
    if resume_stage > len(stages):
        raise SystemExit(f"Error: VALIDATION_RESUME_STAGE={resume_stage} exceeds stage_count={len(stages)}")
    return stages[resume_stage - 1 :], resume_stage


def _prepare_validation_dirs(paths: _ValidationPaths, resume_stage: int) -> None:
    if resume_stage == 1 and paths.snapshot_root.exists():
        shutil.rmtree(paths.snapshot_root)
    paths.snapshot_root.mkdir(parents=True, exist_ok=True)
    paths.history_path.parent.mkdir(parents=True, exist_ok=True)
    paths.train_stage_root.mkdir(parents=True, exist_ok=True)
    paths.val_output_root.mkdir(parents=True, exist_ok=True)


def _copy_train_stage_metrics(paths: _ValidationPaths, stage_index: int) -> None:
    train_metrics_file = paths.train_output_root / DATASET_SUMMARY_FILENAME
    if train_metrics_file.exists():
        copy_file_contents(
            train_metrics_file,
            paths.train_stage_root / f"train_stage_{stage_index}_metrics_at_k.json",
        )


def _validation_memory_for_stage(paths: _ValidationPaths, stage_index: int) -> Path:
    _copy_memory_snapshot(paths.train_memory_root, paths.snapshot_root / f"stage_{stage_index}_train")
    validation_memory_root = paths.snapshot_root / f"stage_{stage_index}_eval_work"
    _copy_memory_snapshot(paths.train_memory_root, validation_memory_root)
    return validation_memory_root


def _run_training_stage(
    benchmark: str,
    ctx: MainFlowContext,
    paths: _ValidationPaths,
    stage_index: int,
    train_seen: int,
    stage_json: Path,
) -> None:
    run_phase(
        "train",
        benchmark,
        ctx,
        {
            "TRAIN_DATA_PATH": str(stage_json),
            "TRAIN_OUTPUT_DIR": str(paths.train_output_root),
            "TRAIN_LOG_OUTPUT_DIR": str(ctx.log_root / benchmark / "logs" / "train" / f"stage_{stage_index}"),
            "MAX_SAMPLES_TRAIN": str(train_seen),
        },
    )
    _copy_train_stage_metrics(paths, stage_index)


def _run_validation_rollouts(
    benchmark: str,
    ctx: MainFlowContext,
    paths: _ValidationPaths,
    stage_index: int,
    val_stage_dir: Path,
    validation_memory_root: Path,
) -> None:
    run_phase(
        "val",
        benchmark,
        ctx,
        {
            "VAL_DATA_PATH": str(paths.val_file),
            "VAL_OUTPUT_DIR": str(val_stage_dir),
            "VAL_LOG_OUTPUT_DIR": str(ctx.log_root / benchmark / "logs" / "val" / f"val_stage_{stage_index}"),
            "MAX_SAMPLES_VAL": str(_count_json_items(paths.val_file)),
            "MEMORY_ROOT_OVERRIDE": str(validation_memory_root),
        },
    )


def _apply_stage_feedback(
    val_stage_dir: Path,
    validation_memory_root: Path,
    stage_index: int,
    val_k: int,
) -> None:
    apply_validation_feedback(
        metrics_dir=val_stage_dir,
        memory_root=validation_memory_root,
        stage=str(stage_index),
        output_json=val_stage_dir / "feedback.json",
        k=val_k,
        min_validation_uses_to_disable=3,
        disable_average_threshold=0.01,
        disable_pass_threshold=0.0,
    )


def _run_validation_reflection(
    ctx: MainFlowContext,
    paths: _ValidationPaths,
    val_stage_dir: Path,
    validation_memory_root: Path,
    stage_index: int,
    val_k: int,
) -> None:
    if not bool(_ctx_setting(ctx, "ablation.modules.validation_reflection", True)):
        return
    if not bool(_validation_setting(ctx, "reflection", False)):
        return
    reflection_json = val_stage_dir / "reflection.json"
    reflect_validation_failures(
        metrics_dir=val_stage_dir,
        memory_root=validation_memory_root,
        data_file=paths.val_file,
        stage=str(stage_index),
        output_json=reflection_json,
        skill_library=validation_memory_root / "skill_library.anchor",
        k=val_k,
        max_samples=int(_validation_setting(ctx, "reflection_max_samples", 4)),
        min_average=0.25,
    )
    if _reflection_has_writeback(reflection_json):
        print("Validation reflection already rebuilt skills during write-back.")
        calibrate_validation_status(
            memory_root=validation_memory_root,
            output_json=val_stage_dir / "post_reflection_status_calibration.json",
        )
    else:
        print("Skipping skill rebuild after validation reflection: no new reflection operations were written.")


def _record_stage_history(
    paths: _ValidationPaths,
    val_stage_dir: Path,
    validation_memory_root: Path,
    stage_index: int,
    train_seen: int,
    val_k: int,
) -> None:
    record_validation_stage(
        split="val",
        stage=str(stage_index),
        stage_index=stage_index,
        train_samples_seen=train_seen,
        metrics_dir=val_stage_dir,
        memory_root=validation_memory_root,
        history_path=paths.history_path,
        output_json=val_stage_dir / "summary.json",
        k=val_k,
    )


def _run_one_validation_stage(
    benchmark: str,
    ctx: MainFlowContext,
    paths: _ValidationPaths,
    row: dict[str, Any],
) -> None:
    stage_index = int(row["stage_index"])
    train_seen = int(row["train_samples_seen"])
    stage_json = Path(row["resolved_path"])
    print("=" * 70)
    print(f"Validation stage {stage_index}: train_samples_seen={train_seen}")
    print("=" * 70)

    val_stage_dir = paths.val_output_root / f"val_stage_{stage_index}"
    val_stage_dir.mkdir(parents=True, exist_ok=True)
    calibrate_validation_status(
        memory_root=paths.train_memory_root,
        output_json=val_stage_dir / "status_calibration.json",
    )

    _run_training_stage(benchmark, ctx, paths, stage_index, train_seen, stage_json)
    validation_memory_root = _validation_memory_for_stage(paths, stage_index)
    val_k = int(_rollout_setting(ctx, "per_sample", 4))
    _run_validation_rollouts(benchmark, ctx, paths, stage_index, val_stage_dir, validation_memory_root)
    _apply_stage_feedback(val_stage_dir, validation_memory_root, stage_index, val_k)
    _run_validation_reflection(ctx, paths, val_stage_dir, validation_memory_root, stage_index, val_k)
    _record_stage_history(paths, val_stage_dir, validation_memory_root, stage_index, train_seen, val_k)
    _copy_memory_snapshot(validation_memory_root, paths.snapshot_root / f"stage_{stage_index}")


def _finalize_validation_run(
    benchmark: str,
    ctx: MainFlowContext,
    paths: _ValidationPaths,
) -> None:
    write_validation_report(
        run_root=paths.benchmark_output_root,
        output_dir=paths.analysis_output_root,
        generate_figures=False,
    )
    if bool(_validation_setting(ctx, "final_select_best", False)):
        _copy_memory_snapshot(paths.train_memory_root, paths.snapshot_root / "final_train_memory")
        output_memory_root = paths.train_memory_root
    else:
        output_memory_root = Path(
            os.environ.get("VALIDATION_BEST_MEMORY_ROOT")
            or ctx.output_root / benchmark / "memory" / "best_validation_memory"
        )
    select_best(
        history_path=paths.history_path,
        snapshot_root=paths.snapshot_root,
        output_memory_root=output_memory_root,
        output_json=paths.selected_json,
    )


def run_train_with_validation(benchmark: str, ctx: MainFlowContext) -> None:
    paths = _validation_paths(benchmark, ctx)
    stages, resume_stage = _validation_stages_to_run(paths)
    _prepare_validation_dirs(paths, resume_stage)
    for row in stages:
        _run_one_validation_stage(benchmark, ctx, paths, row)
    _finalize_validation_run(benchmark, ctx, paths)
