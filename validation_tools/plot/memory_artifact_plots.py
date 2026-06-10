"""Memory-library, task-outcome, and artifact-distribution plots."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from utils.artifacts import SAMPLE_SUMMARY_FILENAME, read_json
from .base import (
    COLORS,
    PALETTE,
    apply_paper_style,
    _EmbeddingRecord,
    _clean_label,
    _compact_legend,
    _fallback_task_label_from_id,
    _label_panel,
    _primary_task,
    _save_figure,
)
from .embedding_plots import _plot_memory_embedding_space


def _load_memory_experiences(memory_root: Path) -> dict[str, dict[str, Any]]:
    path = memory_root / "experiences.json"
    if not path.exists():
        return {}
    data = read_json(path)
    if isinstance(data, dict):
        payload = data.get("experiences", data)
        if isinstance(payload, dict):
            return {str(k): v for k, v in payload.items() if isinstance(v, dict)}
        if isinstance(payload, list):
            return {
                str(item.get("id") or item.get("experience_id") or idx): item
                for idx, item in enumerate(payload)
                if isinstance(item, dict)
            }
    if isinstance(data, list):
        return {
            str(item.get("id") or item.get("experience_id") or idx): item
            for idx, item in enumerate(data)
            if isinstance(item, dict)
        }
    return {}


def _load_memory_skills(memory_root: Path) -> dict[str, dict[str, Any]]:
    path = memory_root / "skill_index.json"
    if not path.exists():
        return {}
    data = read_json(path)
    payload = data.get("skills", data) if isinstance(data, dict) else {}
    if not isinstance(payload, dict):
        return {}
    return {str(k): v for k, v in payload.items() if isinstance(v, dict)}


def _collect_embedding_records(
    cache_dir: Path,
    metadata_by_id: dict[str, dict[str, Any]],
    entity_type: str,
) -> list[_EmbeddingRecord]:
    """Read cached embeddings saved either as id->vector dicts or dense matrices."""
    if not cache_dir.exists() or not metadata_by_id:
        return []

    records: dict[str, _EmbeddingRecord] = {}
    metadata_files = sorted(cache_dir.glob("*_metadata.json"), key=lambda p: p.stat().st_mtime)
    for meta_path in metadata_files:
        npy_path = meta_path.with_name(meta_path.name.replace("_metadata.json", ".npy"))
        if not npy_path.exists():
            continue
        meta = read_json(meta_path)
        ids = [str(item) for item in meta.get("experience_ids", [])]
        if not ids:
            continue
        try:
            matrix = np.load(npy_path, allow_pickle=True)
        except Exception:
            continue

        if isinstance(matrix, np.ndarray) and matrix.shape == () and isinstance(matrix.item(), dict):
            vector_by_id = {str(k): v for k, v in matrix.item().items()}
        elif isinstance(matrix, dict):
            vector_by_id = {str(k): v for k, v in matrix.items()}
        elif isinstance(matrix, np.ndarray) and matrix.ndim == 2 and matrix.shape[0] == len(ids):
            vector_by_id = dict(zip(ids, matrix))
        else:
            continue

        for item_id in ids:
            if item_id not in metadata_by_id or item_id not in vector_by_id:
                continue
            records[item_id] = _EmbeddingRecord(
                item_id=item_id,
                vector=np.asarray(vector_by_id[item_id], dtype=float),
                metadata=metadata_by_id[item_id],
                entity_type=entity_type,
            )
    return list(records.values())


def _load_sample_outcomes(run_dir: Path, data_file: Path | None = None) -> list[dict[str, Any]]:
    """Load per-sample success/failure and average score from metrics_sample.json files."""
    task_by_id = {}
    if data_file and data_file.exists():
        data = read_json(data_file)
        if isinstance(data, list):
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    continue
                question_id = str(item.get("doc_id") or item.get("question_id") or f"sample_{idx}")
                task_by_id[question_id] = str(item.get("task") or _fallback_task_label_from_id(question_id))

    outcomes = []
    if not run_dir or not run_dir.exists():
        return outcomes
    for metric_path in sorted(run_dir.glob(f"*/{SAMPLE_SUMMARY_FILENAME}")):
        try:
            metric = read_json(metric_path)
        except Exception:
            continue
        question_id = str(metric.get("question_id") or metric_path.parent.name)
        accuracies = [float(value) for value in metric.get("accuracies", []) if value is not None]
        success = any(value >= 1.0 for value in accuracies)
        average = sum(accuracies) / len(accuracies) if accuracies else 0.0
        outcomes.append(
            {
                "question_id": question_id,
                "task": task_by_id.get(question_id, _fallback_task_label_from_id(question_id)),
                "success": success,
                "average": average,
            }
        )
    return outcomes


def _top_items(counter: Counter, max_items: int = 12) -> tuple[list[str], list[float]]:
    items = counter.most_common(max_items)
    return [item[0] for item in items][::-1], [float(item[1]) for item in items][::-1]


def _barh_counter(ax, counter: Counter, *, color: str, xlabel: str, max_items: int = 12) -> None:
    labels, values = _top_items(counter, max_items)
    ax.barh([_clean_label(label) for label in labels], values, color=color, edgecolor="white", linewidth=0.55)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2, linewidth=0.45)


def _stacked_task_counts(
    ax,
    by_task: dict[str, Counter],
    keys: list[str],
    *,
    xlabel: str,
    max_items: int = 12,
    show_legend: bool = True,
) -> None:
    totals = {task: sum(counter[key] for key in keys) for task, counter in by_task.items()}
    tasks = [task for task, _ in sorted(totals.items(), key=lambda item: item[1], reverse=True)[:max_items]][::-1]
    left = np.zeros(len(tasks))
    colors = {"success": COLORS["success"], "failure": COLORS["failure"]}
    for idx, key in enumerate(keys):
        values = np.asarray([by_task[task][key] for task in tasks], dtype=float)
        ax.barh(
            [_clean_label(task) for task in tasks],
            values,
            left=left,
            color=colors.get(key, PALETTE[idx % len(PALETTE)]),
            edgecolor="white",
            linewidth=0.55,
            label=_clean_label(key),
        )
        left += values
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2, linewidth=0.45)
    if show_legend:
        _compact_legend(ax, ncol=min(2, len(keys)), loc="lower right")


def _plot_memory_artifact_overview(
    experiences: dict[str, dict[str, Any]],
    skills: dict[str, dict[str, Any]],
    output_dir: Path,
) -> Path | None:
    if not experiences and not skills:
        return None

    apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 4.75), constrained_layout=True)

    kind_counts = Counter(str(record.get("kind", "unknown")) for record in experiences.values())
    _barh_counter(axes[0, 0], kind_counts, color=COLORS["experience"], xlabel="Experiences")
    _label_panel(axes[0, 0], "A")

    by_task: dict[str, Counter] = defaultdict(Counter)
    for record in experiences.values():
        by_task[_primary_task(record.get("task_tags"))][str(record.get("kind", "unknown"))] += 1
    _stacked_task_counts(axes[0, 1], by_task, sorted(kind_counts), xlabel="Experiences")
    _label_panel(axes[0, 1], "B")

    skill_type_counts = Counter(str(record.get("skill_type", "unknown")) for record in skills.values())
    _barh_counter(axes[1, 0], skill_type_counts, color=COLORS["skill"], xlabel="Skills")
    _label_panel(axes[1, 0], "C")

    skill_by_task: dict[str, Counter] = defaultdict(Counter)
    for record in skills.values():
        skill_by_task[_primary_task(record.get("task_tags"))][str(record.get("skill_type", "unknown"))] += 1
    _stacked_task_counts(axes[1, 1], skill_by_task, sorted(skill_type_counts), xlabel="Skills", show_legend=False)
    _label_panel(axes[1, 1], "D")

    return _save_figure(fig, output_dir / "memory_artifact_overview.pdf")


def _plot_task_outcome_overview(
    train_outcomes: list[dict[str, Any]],
    val_outcomes: list[dict[str, Any]],
    output_dir: Path,
) -> Path | None:
    if not train_outcomes and not val_outcomes:
        return None

    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.65), constrained_layout=True)

    for ax, outcomes, label in [(axes[0], train_outcomes, "Train"), (axes[1], val_outcomes, "Validation")]:
        by_task: dict[str, Counter] = defaultdict(Counter)
        for item in outcomes:
            by_task[item["task"]]["success" if item["success"] else "failure"] += 1
        _stacked_task_counts(ax, by_task, ["success", "failure"], xlabel=f"{label} Samples")

    avg_by_task: dict[str, list[float]] = defaultdict(list)
    for item in val_outcomes or train_outcomes:
        avg_by_task[item["task"]].append(float(item.get("average", 0.0) or 0.0))
    averages = {task: sum(values) / len(values) for task, values in avg_by_task.items()}
    tasks = [task for task, _ in sorted(averages.items(), key=lambda item: item[1])[:12]]
    axes[2].barh(
        [_clean_label(task) for task in tasks],
        [averages[task] for task in tasks],
        color=COLORS["average"],
        edgecolor="white",
        linewidth=0.55,
    )
    axes[2].set_xlim(0, 1.02)
    axes[2].set_xlabel("Average@k")
    axes[2].set_ylabel("")
    axes[2].grid(axis="x", alpha=0.2, linewidth=0.45)
    for ax, label in zip(axes, ["A", "B", "C"]):
        _label_panel(ax, label)

    return _save_figure(fig, output_dir / "task_outcome_overview.pdf")


def _infer_run_base(run_root: Path) -> Path:
    """Infer the enclosing run directory from output/<benchmark>."""
    if run_root.name and run_root.parent.name == "output":
        return run_root.parent.parent
    return run_root


def _fixed_split_root(benchmark: str) -> Path:
    split_root = Path(__file__).resolve().parents[2] / "benchmark" / "split" / benchmark
    if not split_root.exists():
        raise FileNotFoundError(
            f"Missing fixed benchmark split: {split_root}. "
            "Run benchmark/build_benchmark_subset.py before generating analysis figures."
        )
    return split_root


def _best_validation_stage(run_root: Path) -> int | None:
    history = run_root / "validation_history.jsonl"
    if not history.exists():
        return None

    best_stage = None
    best_value = -1.0
    with history.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("split") != "val":
                continue
            value = float(row.get("average_at_k", 0.0) or 0.0)
            if value > best_value:
                best_value = value
                best_stage = int(row.get("stage_index") or row.get("stage") or 0)
    return best_stage


def _build_memory_artifact_plots(figures_dir: Path, run_root: Path, seed: int = 42) -> list[Path]:
    """Generate compact memory-distribution and embedding PDFs when artifacts exist."""
    run_base = _infer_run_base(run_root)
    benchmark = run_root.name
    memory_root = run_base / "memory" / benchmark
    split_root = _fixed_split_root(benchmark)
    written: list[Path] = []

    experiences = _load_memory_experiences(memory_root)
    skills = _load_memory_skills(memory_root)
    for path in [_plot_memory_artifact_overview(experiences, skills, figures_dir)]:
        if path is not None:
            written.append(path)

    experience_records = _collect_embedding_records(memory_root / "experiences", experiences, "experience")
    skill_records = _collect_embedding_records(memory_root / "skill_index", skills, "skill")
    written.extend(_plot_memory_embedding_space(experience_records, skill_records, figures_dir, seed))

    train_outcomes = []
    train_dir = run_root / "train"
    if train_dir.exists():
        train_outcomes = _load_sample_outcomes(train_dir, split_root / "train.json")

    val_outcomes = []
    best_stage = _best_validation_stage(run_root)
    if best_stage is not None:
        val_dir = run_root / "val" / f"val_stage_{best_stage}"
        if not val_dir.exists():
            val_dir = run_root / f"val_stage_{best_stage}"
        val_outcomes = _load_sample_outcomes(val_dir, split_root / "val.json")

    path = _plot_task_outcome_overview(train_outcomes, val_outcomes, figures_dir)
    if path is not None:
        written.append(path)
    return written
