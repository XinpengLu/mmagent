"""Embedding-space visualizations for experiences and skills."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .base import PALETTE, _EmbeddingRecord, _primary_task, _save_figure


def _reduce_embeddings(records: list[_EmbeddingRecord], seed: int) -> np.ndarray:
    matrix = np.vstack([record.vector for record in records])
    matrix = StandardScaler().fit_transform(matrix)
    if len(records) <= 3:
        return PCA(n_components=2, random_state=seed).fit_transform(matrix)
    if matrix.shape[1] > 50 and len(records) > 50:
        matrix = PCA(n_components=50, random_state=seed).fit_transform(matrix)
    perplexity = max(2, min(30, (len(records) - 1) // 3))
    return TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
    ).fit_transform(matrix)


def _plot_embedding_space(
    records: list[_EmbeddingRecord],
    output_path: Path,
    label_fn: Callable[[_EmbeddingRecord], str],
    title: str,
    seed: int,
    marker_by_entity: bool = False,
) -> Path | None:
    if len(records) < 2:
        return None
    coords = _reduce_embeddings(records, seed)
    labels = [label_fn(record) for record in records]
    unique_labels = sorted(set(labels))
    color_map = {label: PALETTE[idx % len(PALETTE)] for idx, label in enumerate(unique_labels)}
    marker_map = {"experience": "o", "skill": "^"}

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for label in unique_labels:
        mask = np.array([item == label for item in labels])
        if marker_by_entity:
            for entity_type in sorted({record.entity_type for record in records}):
                entity_mask = np.array([record.entity_type == entity_type for record in records])
                final_mask = mask & entity_mask
                if final_mask.any():
                    ax.scatter(
                        coords[final_mask, 0],
                        coords[final_mask, 1],
                        s=24,
                        marker=marker_map.get(entity_type, "o"),
                        color=color_map[label],
                        alpha=0.82,
                        linewidth=0.3,
                        edgecolor="white",
                        label=f"{label} / {entity_type}",
                    )
        else:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=24,
                color=color_map[label],
                alpha=0.82,
                linewidth=0.3,
                edgecolor="white",
                label=label,
            )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.18, linewidth=0.5)
    if len(unique_labels) <= 12:
        ax.legend(frameon=False, loc="best", ncol=1)
    return _save_figure(fig, output_path)


def _plot_experience_embedding_spaces(
    records: list[_EmbeddingRecord],
    output_dir: Path,
    seed: int,
) -> list[Path]:
    paths = []
    for path in [
        _plot_embedding_space(
            records,
            output_dir / "experience_embedding_by_kind.pdf",
            lambda record: str(record.metadata.get("kind", "unknown")),
            "Experience Embedding Space by Outcome",
            seed,
        ),
        _plot_embedding_space(
            records,
            output_dir / "experience_embedding_by_task.pdf",
            lambda record: _primary_task(record.metadata.get("task_tags")),
            "Experience Embedding Space by Task",
            seed,
        ),
    ]:
        if path is not None:
            paths.append(path)
    return paths


def _plot_skill_embedding_spaces(records: list[_EmbeddingRecord], output_dir: Path, seed: int) -> list[Path]:
    path = _plot_embedding_space(
        records,
        output_dir / "skill_embedding_by_type.pdf",
        lambda record: str(record.metadata.get("skill_type", "unknown")),
        "Skill Embedding Space by Type",
        seed,
    )
    return [path] if path is not None else []


def _plot_memory_embedding_space(
    experience_records: list[_EmbeddingRecord],
    skill_records: list[_EmbeddingRecord],
    output_dir: Path,
    seed: int,
) -> list[Path]:
    path = _plot_embedding_space(
        experience_records + skill_records,
        output_dir / "memory_embedding_by_entity.pdf",
        lambda record: record.entity_type,
        "Experience and Skill Embedding Space",
        seed,
        marker_by_entity=True,
    )
    return [path] if path is not None else []
