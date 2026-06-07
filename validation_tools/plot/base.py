"""Shared helpers for validation report plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List
import matplotlib.pyplot as plt
import numpy as np


PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
]


@dataclass
class _EmbeddingRecord:
    item_id: str
    vector: np.ndarray
    metadata: dict[str, Any]
    entity_type: str


def _stages(rows: List[Dict[str, Any]]) -> List[int]:
    return [int(row.get("stage") or 0) for row in rows]


def _values(rows: List[Dict[str, Any]], key: str) -> List[float]:
    return [float(row.get(key) or 0.0) for row in rows]


def _has_nonzero(rows: Iterable[Dict[str, Any]], key: str) -> bool:
    return any(float(row.get(key) or 0.0) for row in rows)


def _save_figure(fig: Any, path: Path, **savefig_kwargs: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight", **savefig_kwargs}
    fig.savefig(path, **kwargs)
    plt.close(fig)
    return path


def _primary_task(tags: Any, fallback: str = "unknown") -> str:
    """Extract a benchmark-neutral task label from metadata tags."""
    if not isinstance(tags, list):
        return fallback
    normalized = [str(tag).replace("task_", "") for tag in tags]
    for tag in normalized:
        if tag and not tag.startswith("data_source") and tag not in {"general", "reasoning"}:
            return tag
    return normalized[0] if normalized else fallback


def _fallback_task_label_from_id(question_id: str) -> str:
    parts = str(question_id).split("_")
    if len(parts) == 3 and parts[0].isalpha() and parts[-1].isdigit():
        return parts[1]
    return parts[0] if parts else "unknown"
