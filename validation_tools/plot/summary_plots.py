"""Stage-level validation report plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .base import _has_nonzero, _save_figure, _stages, _values


COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#222222",
    "gray": "#666666",
}


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "lines.markersize": 4.5,
        }
    )
    return plt


def _plot_performance(path: Path, rows: List[Dict[str, Any]], target_average: float) -> Path | None:
    if not rows:
        return None

    plt = _setup_matplotlib()
    xs = _stages(rows)
    fig, ax = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)

    ax.plot(xs, _values(rows, "average_at_k"), marker="o", color=COLORS["blue"], label="val average@k")
    ax.plot(xs, _values(rows, "pass_at_k"), marker="s", color=COLORS["orange"], label="val pass@k")
    ax.plot(
        xs,
        _values(rows, "best_average_so_far"),
        linestyle="--",
        color=COLORS["black"],
        label="best val average@k",
    )
    if _has_nonzero(rows, "train_average_at_k"):
        ax.plot(xs, _values(rows, "train_average_at_k"), marker="D", color=COLORS["green"], label="train average@k")
    if _has_nonzero(rows, "train_pass_at_k"):
        ax.plot(xs, _values(rows, "train_pass_at_k"), marker="^", color=COLORS["purple"], label="train pass@k")

    ax.axhline(target_average, color=COLORS["gray"], linestyle=":", linewidth=1.4, label=f"target {target_average:.2f}")
    ax.set_xlabel("training stage")
    ax.set_ylabel("score")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Train/Validation Performance")
    ax.legend(frameon=False, ncol=2)
    return _save_figure(fig, path)


def _plot_efficiency(path: Path, rows: List[Dict[str, Any]]) -> Path | None:
    if not rows:
        return None

    plt = _setup_matplotlib()
    xs = _stages(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)

    ax = axes[0]
    ax.plot(xs, _values(rows, "runtime_seconds_mean"), marker="o", color=COLORS["blue"], label="runtime / sample")
    ax.plot(xs, _values(rows, "api_elapsed_seconds_mean"), marker="s", color=COLORS["green"], label="API elapsed / sample")
    ax.set_xlabel("training stage")
    ax.set_ylabel("seconds")
    ax.set_title("Validation Runtime")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(xs, _values(rows, "api_prompt_tokens_mean"), marker="o", color=COLORS["blue"], label="prompt tokens")
    ax.plot(xs, _values(rows, "api_completion_tokens_mean"), marker="s", color=COLORS["orange"], label="completion tokens")
    ax.plot(xs, _values(rows, "api_total_tokens_mean"), marker="^", color=COLORS["black"], label="total tokens")
    ax.set_xlabel("training stage")
    ax.set_ylabel("tokens / sample")
    ax.set_title("Validation Token Use")
    ax.legend(frameon=False)

    return _save_figure(fig, path)


def _plot_memory(path: Path, rows: List[Dict[str, Any]]) -> Path | None:
    if not rows:
        return None

    plt = _setup_matplotlib()
    xs = _stages(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)

    ax = axes[0]
    ax.plot(xs, _values(rows, "selected_skill_ratio"), marker="o", color=COLORS["blue"], label="selected skill")
    ax.plot(xs, _values(rows, "recovery_skill_ratio"), marker="s", color=COLORS["purple"], label="recovery skill")
    ax.plot(xs, _values(rows, "direct_tool_plan_ratio"), marker="^", color=COLORS["orange"], label="direct tool plan")
    ax.plot(xs, _values(rows, "memory_tool_plan_coverage"), marker="D", color=COLORS["green"], label="tool-plan coverage")
    ax.set_xlabel("training stage")
    ax.set_ylabel("ratio")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Memory Signals Used on Validation")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(xs, _values(rows, "experience_count"), marker="o", color=COLORS["black"], label="all experiences")
    ax.plot(xs, _values(rows, "success_experience_count"), marker="s", color=COLORS["green"], label="success experiences")
    ax.plot(xs, _values(rows, "failure_experience_count"), marker="^", color=COLORS["orange"], label="failure experiences")
    ax.plot(xs, _values(rows, "skill_count"), marker="D", color=COLORS["blue"], label="skills")
    ax.set_xlabel("training stage")
    ax.set_ylabel("count")
    ax.set_title("Memory Library Size")
    ax.legend(frameon=False)

    return _save_figure(fig, path)


def _plot_diagnostics(path: Path, rows: List[Dict[str, Any]], target_average: float) -> Path | None:
    if not rows:
        return None

    plt = _setup_matplotlib()
    xs = _stages(rows)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(xs, _values(rows, "average_at_k"), marker="o", color=COLORS["blue"], label="val average@k")
    ax.plot(xs, _values(rows, "pass_at_k"), marker="s", color=COLORS["orange"], label="val pass@k")
    ax.plot(xs, _values(rows, "best_average_so_far"), linestyle="--", color=COLORS["black"], label="best average")
    if _has_nonzero(rows, "train_average_at_k"):
        ax.plot(xs, _values(rows, "train_average_at_k"), marker="D", color=COLORS["green"], label="train average@k")
    ax.axhline(target_average, color=COLORS["gray"], linestyle=":", linewidth=1.2)
    ax.set_title("Performance")
    ax.set_xlabel("stage")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, ncol=2)

    ax = axes[0, 1]
    ax.plot(xs, _values(rows, "runtime_seconds_mean"), marker="o", color=COLORS["blue"], label="seconds / sample")
    ax2 = ax.twinx()
    ax2.plot(xs, _values(rows, "api_total_tokens_mean"), marker="s", color=COLORS["orange"], label="tokens / sample")
    ax.set_title("Efficiency")
    ax.set_xlabel("stage")
    ax.set_ylabel("seconds")
    ax2.set_ylabel("tokens")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False)

    ax = axes[1, 0]
    ax.plot(xs, _values(rows, "selected_skill_ratio"), marker="o", color=COLORS["blue"], label="selected skill")
    ax.plot(xs, _values(rows, "recovery_skill_ratio"), marker="s", color=COLORS["purple"], label="recovery skill")
    ax.plot(xs, _values(rows, "action_memory_ratio"), marker="^", color=COLORS["green"], label="action memory")
    ax.plot(xs, _values(rows, "direct_tool_plan_ratio"), marker="D", color=COLORS["orange"], label="direct tool plan")
    ax.set_title("Memory Use")
    ax.set_xlabel("stage")
    ax.set_ylabel("ratio")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(xs, _values(rows, "min_task_average_at_k"), marker="o", color=COLORS["orange"], label="weakest task avg")
    ax.plot(xs, _values(rows, "selected_skill_average_at_k"), marker="s", color=COLORS["blue"], label="selected skill avg")
    ax.plot(xs, _values(rows, "recovery_skill_average_at_k"), marker="^", color=COLORS["purple"], label="recovery skill avg")
    ax.set_title("Weak Slices")
    ax.set_xlabel("stage")
    ax.set_ylabel("average@k")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)

    return _save_figure(fig, path)
