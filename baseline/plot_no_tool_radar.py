"""Plot no-tool baseline radar charts for pass@4 and average@4."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "gpt-5.5"
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "output" / DEFAULT_MODEL_NAME / "no_tool"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "figures"
BENCHMARKS = [
    "TIR-Bench",
    "AgentVista",
    "MMSearch-Plus",
    "MMBrowseComp",
    "VisualToolBench",
    "BrowseComp-V3",
    "VisBrowse-Bench",
    "Agentic-MME",
]
METHOD_LABEL = "GPT-5.5 (w/o tool)"
ROTATE_LABEL_180 = {"Agentic-MME", "VisBrowse-Bench"}


def _load_metric(input_root: Path, metric_name: str, k: int = 4) -> list[float]:
    values = []
    for benchmark in BENCHMARKS:
        metrics_path = input_root / benchmark / "test" / "metrics_at_k.json"
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        values.append(float(metrics["k_metrics"][str(k)][metric_name]) * 100.0)
    return values


def _radar_angles(num_axes: int) -> np.ndarray:
    return np.linspace(0, 2 * np.pi, num_axes, endpoint=False)


def _radial_limit(values: list[float]) -> float:
    upper = math.ceil((max(values) + 5.0) / 10.0) * 10.0
    return min(100.0, max(20.0, upper))


def _set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _screen_angle_degrees(angle: float) -> float:
    return 90.0 - math.degrees(angle)


def _label_rotation(angle: float) -> float:
    rotation = _screen_angle_degrees(angle) - 90.0
    if rotation < -90.0:
        rotation += 180.0
    if rotation > 90.0:
        rotation -= 180.0
    return rotation


def _benchmark_label_rotation(benchmark: str, angle: float) -> float:
    rotation = _label_rotation(angle)
    return rotation + 180.0 if benchmark in ROTATE_LABEL_180 else rotation


def _label_radius(angle: float, rmax: float) -> float:
    screen_angle = math.radians(_screen_angle_degrees(angle))
    x = abs(math.cos(screen_angle))
    if x > 0.85:
        return rmax * 1.27
    if x > 0.55:
        return rmax * 1.17
    return rmax * 1.10


def _value_label_offset(angle: float, value: float, rmax: float) -> tuple[float, float]:
    screen_angle = math.radians(_screen_angle_degrees(angle))
    radial = np.array([math.cos(screen_angle), math.sin(screen_angle)])
    tangent = np.array([-math.sin(screen_angle), math.cos(screen_angle)])

    radial_points = 13.0
    if value < rmax * 0.30:
        radial_points = 16.0
    if value > rmax * 0.86:
        radial_points = -14.0

    side = 1.0 if math.sin(angle) >= 0 else -1.0
    offset = radial * radial_points + tangent * side * 7.0
    return float(offset[0]), float(offset[1])


def _plot_radar_panel(ax, values: list[float], title: str) -> None:
    angles = _radar_angles(len(BENCHMARKS))
    closed_angles = np.concatenate([angles, angles[:1]])
    closed_values = np.array(values + values[:1])
    rmax = _radial_limit(values)
    rticks = np.linspace(0, rmax, 5)[1:]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, rmax)

    color = "#2f855a"
    fill_color = "#9ae6b4"
    ax.plot(
        closed_angles,
        closed_values,
        color=color,
        linewidth=2.4,
        marker="o",
        markersize=4.8,
        label=METHOD_LABEL,
    )
    ax.fill(closed_angles, closed_values, color=fill_color, alpha=0.22)

    ax.set_xticks(angles)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    for angle, benchmark in zip(angles, BENCHMARKS):
        ax.text(
            angle,
            _label_radius(angle, rmax),
            benchmark,
            ha="center",
            va="center",
            rotation=_benchmark_label_rotation(benchmark, angle),
            rotation_mode="anchor",
            fontsize=12,
            fontweight="semibold",
            color="#1f2937",
            clip_on=False,
        )
    ax.set_yticks(rticks)
    ax.set_yticklabels([f"{tick:.0f}" for tick in rticks], color="#64748b", fontsize=9)
    ax.set_rlabel_position(90)
    ax.grid(color="#d9dee8", linewidth=0.9, alpha=0.78)
    ax.spines["polar"].set_color("#d1d5db")
    ax.spines["polar"].set_linewidth(0.9)

    for angle, value in zip(angles, values):
        ax.annotate(
            f"{value:.1f}",
            xy=(angle, value),
            xytext=_value_label_offset(angle, value, rmax),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=color,
            bbox={
                "boxstyle": "round,pad=0.08",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
        )

    ax.text(
        0.5,
        -0.19,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#111827",
    )


def _plot_combined_radar(charts: list[tuple[str, list[float]]], output_path: Path) -> None:
    _set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.35), subplot_kw={"projection": "polar"})
    for ax, (title, values) in zip(axes, charts):
        _plot_radar_panel(ax, values, title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.88, bottom=0.22, wspace=0.34)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    charts = [
        ("Average@4 (%)", _load_metric(args.input_root, "average_at_k", k=4)),
        ("Pass@4 (%)", _load_metric(args.input_root, "pass_at_k", k=4)),
    ]
    output_path = args.output_dir / "no_tool_average_pass_at_4_radar.pdf"
    _plot_combined_radar(charts, output_path)
    print(f"Saved {output_path}")
    print(f"Saved {output_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
