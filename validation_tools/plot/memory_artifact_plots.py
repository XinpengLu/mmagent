"""Memory-library, task-outcome, and artifact-distribution plots."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from utils.artifacts import SAMPLE_SUMMARY_FILENAME, read_json
from .base import _EmbeddingRecord, PALETTE, _fallback_task_label_from_id, _primary_task, _save_figure
from .embedding_plots import (
    _plot_experience_embedding_spaces,
    _plot_memory_embedding_space,
    _plot_skill_embedding_spaces,
)


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


def _save_bar(
    labels: list[str],
    values_: list[float],
    output_path: Path,
    title: str,
    ylabel: str,
    color: str = "#0072B2",
) -> Path:
    fig, ax = plt.subplots(figsize=(max(4.2, 0.42 * len(labels)), 3.2))
    ax.bar(labels, values_, color=color, edgecolor="white", linewidth=0.6)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    return _save_figure(fig, output_path)


def _save_stacked_bar(
    labels: list[str],
    stacks: dict[str, list[float]],
    output_path: Path,
    title: str,
    ylabel: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(max(4.4, 0.48 * len(labels)), 3.3))
    bottom = np.zeros(len(labels))
    for idx, (name, values_) in enumerate(stacks.items()):
        ax.bar(
            labels,
            values_,
            bottom=bottom,
            label=name,
            color=PALETTE[idx % len(PALETTE)],
            edgecolor="white",
            linewidth=0.6,
        )
        bottom += np.asarray(values_)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, ncol=min(3, len(stacks)))
    return _save_figure(fig, output_path)


def _plot_experience_counts(experiences: dict[str, dict[str, Any]], output_dir: Path) -> list[Path]:
    if not experiences:
        return []
    written: list[Path] = []
    kind_counts = Counter(str(record.get("kind", "unknown")) for record in experiences.values())
    labels = sorted(kind_counts)
    written.append(
        _save_bar(
            labels,
            [kind_counts[label] for label in labels],
            output_dir / "experience_kind_counts.pdf",
            "Experience Outcomes",
            "Number of experiences",
            color="#0072B2",
        )
    )

    by_task: dict[str, Counter] = defaultdict(Counter)
    for record in experiences.values():
        task = _primary_task(record.get("task_tags"))
        by_task[task][str(record.get("kind", "unknown"))] += 1
    tasks = sorted(by_task)
    kinds = sorted(kind_counts)
    written.append(
        _save_stacked_bar(
            tasks,
            {kind: [by_task[task][kind] for task in tasks] for kind in kinds},
            output_dir / "experience_kind_by_task.pdf",
            "Experience Outcomes by Task",
            "Number of experiences",
        )
    )
    return written


def _plot_skill_counts(skills: dict[str, dict[str, Any]], output_dir: Path) -> list[Path]:
    if not skills:
        return []
    written: list[Path] = []
    type_counts = Counter(str(record.get("skill_type", "unknown")) for record in skills.values())
    labels = sorted(type_counts)
    written.append(
        _save_bar(
            labels,
            [type_counts[label] for label in labels],
            output_dir / "skill_type_counts.pdf",
            "Skill Types",
            "Number of skills",
            color="#009E73",
        )
    )

    by_task: dict[str, Counter] = defaultdict(Counter)
    for record in skills.values():
        task = _primary_task(record.get("task_tags"))
        by_task[task][str(record.get("skill_type", "unknown"))] += 1
    tasks = sorted(by_task)
    types = sorted(type_counts)
    written.append(
        _save_stacked_bar(
            tasks,
            {kind: [by_task[task][kind] for task in tasks] for kind in types},
            output_dir / "skill_type_by_task.pdf",
            "Skill Types by Task",
            "Number of skills",
        )
    )
    return written


def _plot_task_outcomes(outcomes: list[dict[str, Any]], output_dir: Path, prefix: str) -> list[Path]:
    if not outcomes:
        return []
    written: list[Path] = []
    by_task: dict[str, Counter] = defaultdict(Counter)
    avg_by_task: dict[str, list[float]] = defaultdict(list)
    for item in outcomes:
        task = item["task"]
        by_task[task]["success" if item["success"] else "failure"] += 1
        avg_by_task[task].append(float(item.get("average", 0.0) or 0.0))
    tasks = sorted(by_task)
    written.append(
        _save_stacked_bar(
            tasks,
            {
                "success": [by_task[task]["success"] for task in tasks],
                "failure": [by_task[task]["failure"] for task in tasks],
            },
            output_dir / f"{prefix}_task_success_failure_counts.pdf",
            f"{prefix.replace('_', ' ').title()} Task Outcomes",
            "Number of samples",
        )
    )

    averages = [sum(avg_by_task[task]) / len(avg_by_task[task]) for task in tasks]
    fig, ax = plt.subplots(figsize=(max(4.4, 0.48 * len(tasks)), 3.3))
    ax.bar(tasks, averages, color="#0072B2", edgecolor="white", linewidth=0.6)
    ax.set_title(f"{prefix.replace('_', ' ').title()} Average Score by Task")
    ax.set_ylabel("mean sample average@k")
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    path = output_dir / f"{prefix}_task_average_scores.pdf"
    written.append(_save_figure(fig, path))
    return written


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


def _build_memory_artifact_plots(figures_dir: Path, run_root: Path, seed: int = 42) -> list[Path]:
    """Generate memory-distribution and embedding PDFs when memory artifacts exist."""
    run_base = _infer_run_base(run_root)
    benchmark = run_root.name
    memory_root = run_base / "memory" / benchmark
    split_root = _fixed_split_root(benchmark)
    written: list[Path] = []

    experiences = _load_memory_experiences(memory_root)
    skills = _load_memory_skills(memory_root)
    if experiences:
        written.extend(_plot_experience_counts(experiences, figures_dir))
    if skills:
        written.extend(_plot_skill_counts(skills, figures_dir))

    experience_records = _collect_embedding_records(memory_root / "experiences", experiences, "experience")
    skill_records = _collect_embedding_records(memory_root / "skill_index", skills, "skill")
    written.extend(_plot_experience_embedding_spaces(experience_records, figures_dir, seed))
    written.extend(_plot_skill_embedding_spaces(skill_records, figures_dir, seed))
    written.extend(_plot_memory_embedding_space(experience_records, skill_records, figures_dir, seed))

    train_dir = run_root / "train"
    if train_dir.exists():
        written.extend(_plot_task_outcomes(_load_sample_outcomes(train_dir, split_root / "train.json"), figures_dir, "train"))

    best_stage = _best_validation_stage(run_root)
    if best_stage is not None:
        val_dir = run_root / "val" / f"val_stage_{best_stage}"
        if not val_dir.exists():
            val_dir = run_root / f"val_stage_{best_stage}"
        written.extend(_plot_task_outcomes(_load_sample_outcomes(val_dir, split_root / "val.json"), figures_dir, f"val_stage_{best_stage}"))
    return written


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
