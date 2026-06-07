"""Structured validation metrics and publication-style plots."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

from .history import read_history
from .plot import build_all_plots
from utils.artifacts import SAMPLE_SUMMARY_FILENAME, read_json, write_json


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    return mean(vals) if vals else 0.0


def _int_text(value: object) -> int | None:
    text = str(value).strip()
    return int(text) if text.isdigit() else None


def _select_k_metric(k_metrics: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(k_metrics, dict) or not k_metrics:
        return {}
    numeric_keys = [value for key in k_metrics if (value := _int_text(key)) is not None]
    if numeric_keys:
        selected_key = str(max(numeric_keys))
        selected = k_metrics.get(selected_key) or k_metrics.get(max(numeric_keys))
    else:
        selected = next(iter(k_metrics.values()), {})
    return selected if isinstance(selected, dict) else {}


def _outcome_value(outcome_group: Dict[str, Any], metric_prefix: str) -> Any:
    if not isinstance(outcome_group, dict):
        return None
    candidates = []
    for key, value in outcome_group.items():
        if not str(key).startswith(f"{metric_prefix}@"):
            continue
        suffix = str(key).split("@", 1)[1] if "@" in str(key) else ""
        index = _int_text(suffix)
        if index is not None:
            candidates.append((index, value))
    if candidates:
        return sorted(candidates, key=lambda item: item[0])[-1][1]
    return outcome_group.get(f"{metric_prefix}@4")


def _stage_index_from_dir(path: Path) -> int | None:
    match = re.match(r"val_stage_(\d+)$", path.name)
    return int(match.group(1)) if match else None


def _sample_summaries(stage_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for metrics_path in sorted(stage_dir.glob(f"*/{SAMPLE_SUMMARY_FILENAME}")):
        row = read_json(metrics_path)
        if isinstance(row, dict):
            row["_sample_id"] = metrics_path.parent.name
            rows.append(row)
    return rows


def _efficiency_for_stage(stage_dir: Path) -> Dict[str, Any]:
    rows = _sample_summaries(stage_dir)
    return {
        "sample_count": len(rows),
        "runtime_seconds_mean": _mean(row.get("runtime_seconds_mean") for row in rows),
        "runtime_seconds_sum": sum(float(row.get("runtime_seconds_sum") or 0.0) for row in rows),
        "api_elapsed_seconds_mean": _mean(row.get("api_elapsed_seconds_mean") for row in rows),
        "api_total_tokens_mean": _mean(row.get("api_total_tokens_mean") for row in rows),
        "api_total_tokens_sum": sum(float(row.get("api_total_tokens_sum") or 0.0) for row in rows),
        "api_prompt_tokens_mean": _mean(row.get("api_prompt_tokens_mean") for row in rows),
        "api_completion_tokens_mean": _mean(row.get("api_completion_tokens_mean") for row in rows),
        "estimated_context_tokens_mean": _mean(row.get("estimated_context_tokens_mean") for row in rows),
        "turn_count_mean": _mean(row.get("turn_count_mean") for row in rows),
        "tool_call_count_mean": _mean(row.get("tool_call_count_mean") for row in rows),
        "api_call_count_mean": _mean(row.get("api_call_count_mean") for row in rows),
    }


def _train_metrics_for_stage(run_root: Path, stage_idx: int) -> Dict[str, Any]:
    path = run_root / "train" / "stages" / f"train_stage_{stage_idx}_metrics_at_k.json"
    if not path.exists():
        path = run_root / f"train_stage_{stage_idx}_metrics_at_k.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    k_metrics = payload.get("k_metrics") if isinstance(payload, dict) else {}
    selected = _select_k_metric(k_metrics)
    if not isinstance(selected, dict):
        return {}
    return {
        "train_num_samples": payload.get("num_samples"),
        "train_average_at_k": selected.get("average_at_k"),
        "train_pass_at_k": selected.get("pass_at_k"),
    }


def _validation_stage_dirs(run_root: Path) -> List[Path]:
    val_root = run_root / "val"
    stage_dirs = list(val_root.glob("val_stage_*")) if val_root.exists() else []
    return stage_dirs or list(run_root.glob("val_stage_*"))


def _stage_metric_row(run_root: Path, stage_dir: Path, history_rows: Dict[int, Dict[str, Any]]) -> Dict[str, Any] | None:
    stage_idx = _stage_index_from_dir(stage_dir)
    if stage_idx is None:
        return None
    hist = history_rows.get(stage_idx, {})
    usage = hist.get("memory_action_usage") or {}
    outcome = usage.get("outcome_by_signal") or {}
    tool_usage = hist.get("tool_usage") or {}
    mem_stats = hist.get("memory_stats") or {}
    experience_kind_counts = mem_stats.get("experience_kind_counts") or {}
    selected_skill = outcome.get("selected_skill") or {}
    recovery_skill = outcome.get("recovery_skill") or {}
    direct_tool_plan = outcome.get("direct_tool_plan") or {}
    no_selected_skill = outcome.get("no_selected_skill") or {}
    row = {
        "stage": stage_idx,
        "train_samples_seen": hist.get("train_samples_seen", stage_idx),
        "average_at_k": hist.get("average_at_k"),
        "pass_at_k": hist.get("pass_at_k"),
        "stability_gap": (
            float(hist.get("pass_at_k", 0.0) or 0.0)
            - float(hist.get("average_at_k", 0.0) or 0.0)
        ),
        "min_task_average_at_k": hist.get("min_task_average_at_k"),
        "weakest_task": hist.get("weakest_task"),
        "selected_skill_ratio": usage.get("selected_skill_ratio"),
        "recovery_skill_ratio": usage.get("recovery_skill_ratio"),
        "unverified_recovery_skill_ratio": usage.get("unverified_recovery_skill_ratio"),
        "memory_tool_plan_coverage": usage.get("memory_tool_plan_coverage"),
        "action_memory_ratio": usage.get("action_memory_ratio"),
        "direct_tool_plan_ratio": usage.get("direct_tool_plan_ratio"),
        "selected_skill_average_at_k": _outcome_value(selected_skill, "average"),
        "selected_skill_pass_at_k": _outcome_value(selected_skill, "pass"),
        "recovery_skill_average_at_k": _outcome_value(recovery_skill, "average"),
        "recovery_skill_pass_at_k": _outcome_value(recovery_skill, "pass"),
        "direct_tool_plan_average_at_k": _outcome_value(direct_tool_plan, "average"),
        "direct_tool_plan_pass_at_k": _outcome_value(direct_tool_plan, "pass"),
        "no_selected_skill_average_at_k": _outcome_value(no_selected_skill, "average"),
        "no_selected_skill_pass_at_k": _outcome_value(no_selected_skill, "pass"),
        "with_tool_ratio": tool_usage.get("with_tool_ratio"),
        "avg_tool_calls_per_rollout": tool_usage.get("avg_tool_calls_per_rollout"),
        "experience_count": mem_stats.get("experience_count"),
        "success_experience_count": experience_kind_counts.get("success"),
        "failure_experience_count": experience_kind_counts.get("failure"),
        "validation_skill_count": mem_stats.get("validation_skill_count"),
        "disabled_skill_count": mem_stats.get("disabled_skill_count"),
        "validation_quality_avg": mem_stats.get("validation_quality_avg"),
        "skill_count": mem_stats.get("skill_count"),
    }
    row.update(_efficiency_for_stage(stage_dir))
    row.update(_train_metrics_for_stage(run_root, stage_idx))
    return row


def _annotate_best_so_far(stage_rows: List[Dict[str, Any]]) -> None:
    best_average = 0.0
    best_pass = 0.0
    for row in stage_rows:
        best_average = max(best_average, float(row.get("average_at_k") or 0.0))
        best_pass = max(best_pass, float(row.get("pass_at_k") or 0.0))
        row["best_average_so_far"] = round(best_average, 6)
        row["best_pass_so_far"] = round(best_pass, 6)


def _collect_stage_metrics(run_root: Path) -> List[Dict[str, Any]]:
    history_path = run_root / "validation_history.jsonl"
    history_rows = {
        int(row.get("stage_index", 0) or 0): row
        for row in read_history(history_path)
        if row.get("split") == "val"
    }

    stage_rows: List[Dict[str, Any]] = []
    for stage_dir in sorted(_validation_stage_dirs(run_root)):
        row = _stage_metric_row(run_root, stage_dir, history_rows)
        if row is not None:
            stage_rows.append(row)
    stage_rows.sort(key=lambda row: int(row.get("stage", 0) or 0))
    _annotate_best_so_far(stage_rows)
    return stage_rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_validation_report(
    *,
    run_root: Path,
    output_dir: Path | None = None,
    target_average: float = 0.70,
) -> None:
    out_dir = output_dir or (run_root / "analysis")
    rows = _collect_stage_metrics(run_root)
    metrics_dir = out_dir / "metrics"
    figures_dir = out_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    write_json(metrics_dir / "stage_metrics.json", {"stages": rows}, indent=2)
    _write_csv(metrics_dir / "stage_metrics.csv", rows)
    if rows:
        build_all_plots(figures_dir, rows, target_average, run_root)

    best = max(
        rows,
        key=lambda row: (float(row.get("average_at_k") or 0.0), float(row.get("pass_at_k") or 0.0)),
        default={},
    )
    print(json.dumps({"output_dir": str(out_dir), "metrics_dir": str(metrics_dir), "figures_dir": str(figures_dir), "best": best}, ensure_ascii=False, indent=2))
