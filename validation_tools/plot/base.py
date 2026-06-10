"""Shared helpers for validation report plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Colorblind-safe, print-friendly palette inspired by Okabe-Ito with a few
# muted extensions for dense multi-panel figures.
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # sky
    "#F0E442",  # yellow
    "#6C6C6C",  # gray
    "#1B1B1B",  # near black
]

COLORS = {
    "average": PALETTE[0],
    "pass": PALETTE[1],
    "train": PALETTE[2],
    "best": PALETTE[8],
    "skill": PALETTE[3],
    "experience": PALETTE[0],
    "success": PALETTE[2],
    "failure": PALETTE[1],
    "recovery": PALETTE[3],
    "tool": PALETTE[4],
    "tokens": PALETTE[1],
    "runtime": PALETTE[0],
    "weak": PALETTE[4],
    "neutral": PALETTE[7],
}


def apply_paper_style() -> None:
    """Apply one compact serif style across all validation report figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 9.2,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.45,
            "lines.linewidth": 1.85,
            "lines.markersize": 4.2,
            "patch.linewidth": 0.55,
            "mathtext.fontset": "stix",
        }
    )


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
    apply_paper_style()
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.02, **savefig_kwargs}
    fig.savefig(path, **kwargs)
    plt.close(fig)
    return path


def _compact_legend(ax: Any, *, ncol: int = 1, loc: str = "best", **kwargs: Any) -> None:
    ax.legend(frameon=False, ncol=ncol, loc=loc, handlelength=1.35, columnspacing=0.75, **kwargs)


def _label_panel(ax: Any, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.5},
    )


def _clean_label(value: Any) -> str:
    return str(value).replace("_", " ").replace("task ", "").title()


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
