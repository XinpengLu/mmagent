"""Compact stage-level validation report plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .base import (
    COLORS,
    apply_paper_style,
    _compact_legend,
    _label_panel,
    _save_figure,
    _stages,
    _values,
)


def _plt():
    apply_paper_style()
    import matplotlib.pyplot as plt

    return plt


def _stage_ticks(ax: Any, xs: list[int]) -> None:
    ax.set_xticks(xs)
    if len(xs) == 1:
        ax.set_xlim(xs[0] - 0.5, xs[0] + 0.5)


def _finish_score_axis(ax: Any, xs: list[int], xlabel: str = "Step") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.02)
    _stage_ticks(ax, xs)


def _tight_score_range(ax: Any, values: list[float]) -> None:
    low = min(values)
    high = max(values)
    pad = max((high - low) * 0.15, 0.03)
    ax.set_ylim(max(0.0, low - pad), min(1.02, high + pad))


def _plot_efficiency(path: Path, rows: List[Dict[str, Any]]) -> Path | None:
    if not rows:
        return None

    plt = _plt()
    xs = _stages(rows)
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.35), constrained_layout=True)

    ax = axes[0]
    ax.plot(xs, _values(rows, "runtime_seconds_mean"), marker="o", color=COLORS["runtime"], label="wall time")
    ax.plot(xs, _values(rows, "api_elapsed_seconds_mean"), marker="s", color=COLORS["train"], label="API time")
    _stage_ticks(ax, xs)
    ax.set_xlabel("Step")
    ax.set_ylabel("Seconds / Sample")
    _compact_legend(ax)
    _label_panel(ax, "A")

    ax = axes[1]
    ax.stackplot(
        xs,
        _values(rows, "api_prompt_tokens_mean"),
        _values(rows, "api_completion_tokens_mean"),
        colors=[COLORS["average"], COLORS["pass"]],
        labels=["prompt", "completion"],
        alpha=0.86,
    )
    ax.plot(xs, _values(rows, "api_total_tokens_mean"), color=COLORS["best"], marker="o", label="total")
    _stage_ticks(ax, xs)
    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens / Sample")
    _compact_legend(ax)
    _label_panel(ax, "B")

    ax = axes[2]
    ax.plot(xs, _values(rows, "turn_count_mean"), marker="o", color=COLORS["skill"], label="turns")
    ax.plot(xs, _values(rows, "tool_call_count_mean"), marker="s", color=COLORS["tool"], label="tool calls")
    ax.plot(xs, _values(rows, "api_call_count_mean"), marker="^", color=COLORS["neutral"], label="API calls")
    _stage_ticks(ax, xs)
    ax.set_xlabel("Step")
    ax.set_ylabel("Count / Sample")
    _compact_legend(ax)
    _label_panel(ax, "C")

    return _save_figure(fig, path)


def _plot_self_evolution_overview(path: Path, rows: List[Dict[str, Any]]) -> Path | None:
    if not rows:
        return None

    plt = _plt()
    xs = _stages(rows)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.0), constrained_layout=True)
    axes = axes.ravel()

    ax = axes[0]
    train_average = _values(rows, "train_average_at_k")
    train_pass = _values(rows, "train_pass_at_k")
    val_average = _values(rows, "average_at_k")
    val_pass = _values(rows, "pass_at_k")
    ax.plot(xs, train_average, marker="o", color=COLORS["train"], label="Train Average")
    ax.plot(xs, train_pass, marker="s", color=COLORS["best"], label="Train Pass")
    ax.plot(xs, val_average, marker="o", color=COLORS["average"], label="Val Average")
    ax.plot(xs, val_pass, marker="s", color=COLORS["pass"], label="Val Pass")
    _finish_score_axis(ax, xs)
    _tight_score_range(ax, train_average + train_pass + val_average + val_pass)
    _compact_legend(ax, ncol=2)

    ax = axes[1]
    ax.plot(xs, _values(rows, "selected_skill_ratio"), marker="o", color=COLORS["average"], label="Selected Skill")
    ax.plot(xs, _values(rows, "recovery_skill_ratio"), marker="s", color=COLORS["skill"], label="Recovery Skill")
    ax.plot(
        xs,
        _values(rows, "action_memory_ratio"),
        marker="^",
        color=COLORS["train"],
        label="Action Memory",
    )
    ax.plot(xs, _values(rows, "direct_tool_plan_ratio"), marker="D", color=COLORS["tool"], label="Direct Tool Plan")
    _stage_ticks(ax, xs)
    ax.set_xlabel("Step")
    ax.set_ylabel("Usage Ratio")
    ax.set_ylim(0.0, 1.02)
    _compact_legend(ax, ncol=2)

    ax = axes[2]
    ax.plot(xs, _values(rows, "experience_count"), marker="o", color=COLORS["experience"], label="Experiences")
    ax.plot(xs, _values(rows, "success_experience_count"), marker="s", color=COLORS["success"], label="Success Exp.")
    ax.plot(xs, _values(rows, "failure_experience_count"), marker="^", color=COLORS["failure"], label="Failure Exp.")
    ax.plot(xs, _values(rows, "skill_count"), marker="D", color=COLORS["skill"], label="Skills")
    _stage_ticks(ax, xs)
    ax.set_xlabel("Step")
    ax.set_ylabel("Count")
    _compact_legend(ax, ncol=2)

    ax = axes[3]
    ax.stackplot(
        xs,
        _values(rows, "api_prompt_tokens_mean"),
        _values(rows, "api_completion_tokens_mean"),
        colors=["#9ECAE1", "#F4A261"],
        labels=["Prompt", "Completion"],
        alpha=0.82,
    )
    ax.plot(xs, _values(rows, "api_total_tokens_mean"), color="#2F2F2F", marker="o", label="Total")
    _stage_ticks(ax, xs)
    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens / Sample")
    _compact_legend(ax, ncol=3)

    return _save_figure(fig, path)
