"""Validation history recording."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .metrics import load_metrics, memory_action_usage, memory_stats, per_task_metrics, tool_usage
from utils.artifacts import write_json


def read_history(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _write_history(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (str(r.get("split", "")), int(r.get("stage_index", 0) or 0), str(r.get("stage", ""))))
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_history_csv(path: Path, rows: List[Dict[str, Any]], k: int) -> None:
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "stage",
                "stage_index",
                "train_samples_seen",
                f"pass@{k}",
                f"average@{k}",
                f"best_pass@{k}",
                f"best_average@{k}",
                "min_task_average",
                "weakest_task",
                "with_tool_ratio",
                "selected_skill_ratio",
                "recovery_skill_ratio",
                "memory_tool_plan_coverage",
                "action_memory_ratio",
                "direct_tool_plan_ratio",
                "experience_count",
                "skill_count",
            ],
        )
        writer.writeheader()
        best_by_split: Dict[str, Dict[str, float]] = {}
        for row in rows:
            split = str(row.get("split") or "")
            current_best = best_by_split.setdefault(split, {"pass": 0.0, "average": 0.0})
            current_best["pass"] = max(current_best["pass"], float(row.get("pass_at_k", 0.0) or 0.0))
            current_best["average"] = max(current_best["average"], float(row.get("average_at_k", 0.0) or 0.0))
            writer.writerow(
                {
                    "split": row.get("split"),
                    "stage": row.get("stage"),
                    "stage_index": row.get("stage_index"),
                    "train_samples_seen": row.get("train_samples_seen"),
                    f"pass@{k}": row.get("pass_at_k"),
                    f"average@{k}": row.get("average_at_k"),
                    f"best_pass@{k}": current_best["pass"],
                    f"best_average@{k}": current_best["average"],
                    "min_task_average": row.get("min_task_average_at_k"),
                    "weakest_task": row.get("weakest_task"),
                    "with_tool_ratio": (row.get("tool_usage") or {}).get("with_tool_ratio"),
                    "selected_skill_ratio": (row.get("memory_action_usage") or {}).get("selected_skill_ratio"),
                    "recovery_skill_ratio": (row.get("memory_action_usage") or {}).get("recovery_skill_ratio"),
                    "memory_tool_plan_coverage": (row.get("memory_action_usage") or {}).get("memory_tool_plan_coverage"),
                    "action_memory_ratio": (row.get("memory_action_usage") or {}).get("action_memory_ratio"),
                    "direct_tool_plan_ratio": (row.get("memory_action_usage") or {}).get("direct_tool_plan_ratio"),
                    "experience_count": (row.get("memory_stats") or {}).get("experience_count"),
                    "skill_count": (row.get("memory_stats") or {}).get("skill_count"),
                }
            )


def record_validation_stage(
    *,
    split: str,
    stage: str,
    stage_index: int,
    train_samples_seen: int,
    metrics_dir: Path,
    memory_root: Path,
    history_path: Path,
    output_json: Path,
    k: int,
) -> None:
    metrics = load_metrics(metrics_dir, k)
    per_task = per_task_metrics(metrics_dir, k)
    usage = tool_usage(metrics_dir)
    mem_stats = memory_stats(memory_root)
    action_usage = memory_action_usage(metrics_dir, memory_root, k)
    if per_task:
        weakest_task, weakest_values = min(per_task.items(), key=lambda kv: kv[1].get("average_at_k", 0.0))
        min_task_average = float(weakest_values.get("average_at_k", 0.0) or 0.0)
        min_task_pass = float(weakest_values.get("pass_at_k", 0.0) or 0.0)
    else:
        weakest_task, min_task_average, min_task_pass = "unknown", 0.0, 0.0

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "split": split,
        "stage": str(stage),
        "stage_index": stage_index,
        "train_samples_seen": train_samples_seen,
        "k": k,
        "num_samples": metrics.get("num_samples"),
        "num_rollouts": metrics.get("num_rollouts"),
        "pass_at_k": metrics["pass_at_k"],
        "average_at_k": metrics["average_at_k"],
        "min_task_average_at_k": min_task_average,
        "min_task_pass_at_k": min_task_pass,
        "weakest_task": weakest_task,
        "metrics_file": metrics.get("metrics_file"),
        "per_task": per_task,
        "tool_usage": usage,
        "memory_action_usage": action_usage,
        "memory_stats": mem_stats,
    }

    rows = read_history(history_path)
    rows = [
        row for row in rows
        if not (row.get("split") == split and str(row.get("stage")) == str(stage))
    ]
    rows.append(entry)
    _write_history(history_path, rows)
    _write_history_csv(history_path, rows, k)

    write_json(output_json, entry, indent=2)

    print(json.dumps(entry, ensure_ascii=False, indent=2))
