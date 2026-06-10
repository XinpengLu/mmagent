"""Embedding-space visualizations for experiences and skills."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .base import COLORS, PALETTE, apply_paper_style, _EmbeddingRecord, _clean_label, _compact_legend, _label_panel, _primary_task, _save_figure


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


def _cluster_ellipse(points: np.ndarray, color: str, ax) -> None:
    if len(points) < 3:
        return
    cov = np.cov(points.T)
    if not np.isfinite(cov).all():
        return
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width, height = 2.35 * np.sqrt(np.maximum(vals, 1e-8))
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ax.add_patch(
        Ellipse(
            points.mean(axis=0),
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            alpha=0.12,
            linewidth=1.0,
            zorder=0,
        )
    )


def _label_by_task(record: _EmbeddingRecord) -> str:
    return _primary_task(record.metadata.get("task_tags"))


def _plot_embedding_panel(ax, records: list[_EmbeddingRecord], coords: np.ndarray, labels: list[str], *, marker_by_entity: bool) -> None:
    unique_labels = sorted(set(labels))
    color_map = {label: PALETTE[idx % len(PALETTE)] for idx, label in enumerate(unique_labels)}
    marker_map = {"experience": "o", "skill": "^"}

    for label in unique_labels:
        mask = np.array([item == label for item in labels])
        _cluster_ellipse(coords[mask], color_map[label], ax)
        if marker_by_entity:
            for entity_type in sorted({record.entity_type for record in records}):
                entity_mask = np.array([record.entity_type == entity_type for record in records])
                final_mask = mask & entity_mask
                if final_mask.any():
                    ax.scatter(
                        coords[final_mask, 0],
                        coords[final_mask, 1],
                        s=28,
                        marker=marker_map.get(entity_type, "o"),
                        color=color_map[label],
                        alpha=0.88,
                        linewidth=0.35,
                        edgecolor="white",
                    )
        else:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=28,
                color=color_map[label],
                alpha=0.88,
                linewidth=0.35,
                edgecolor="white",
            )
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    ax.grid(alpha=0.18, linewidth=0.45)
    if marker_by_entity:
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["experience"], markeredgecolor="white", markersize=6, label="Experience"),
            Line2D([0], [0], marker="^", color="none", markerfacecolor=COLORS["skill"], markeredgecolor="white", markersize=6, label="Skill"),
        ]
        ax.legend(handles=handles, frameon=False, loc="upper right", handlelength=1.0)
    elif len(unique_labels) <= 8:
        _compact_legend(ax, loc="best")


def _plot_memory_embedding_overview(
    experience_records: list[_EmbeddingRecord],
    skill_records: list[_EmbeddingRecord],
    output_dir: Path,
    seed: int,
) -> Path | None:
    records = experience_records + skill_records
    if len(records) < 2:
        return None

    coords = _reduce_embeddings(records, seed)
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.9), constrained_layout=True)

    outcome_labels = [
        str(record.metadata.get("kind") or record.metadata.get("skill_type") or record.entity_type)
        for record in records
    ]
    _plot_embedding_panel(axes[0], records, coords, outcome_labels, marker_by_entity=True)
    _label_panel(axes[0], "A")

    task_labels = [_label_by_task(record) for record in records]
    _plot_embedding_panel(axes[1], records, coords, task_labels, marker_by_entity=True)
    _label_panel(axes[1], "B")

    return _save_figure(fig, output_dir / "memory_embedding_overview.pdf")


def _plot_memory_embedding_space(
    experience_records: list[_EmbeddingRecord],
    skill_records: list[_EmbeddingRecord],
    output_dir: Path,
    seed: int,
) -> list[Path]:
    path = _plot_memory_embedding_overview(experience_records, skill_records, output_dir, seed)
    return [path] if path is not None else []
