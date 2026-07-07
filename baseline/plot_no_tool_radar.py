"""Plot baseline radar charts grouped by reasoning model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "baseline_figures"
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
DEFAULT_MODELS = ["gpt-5.5", "qwen3.6-chat", "deepseek-v4-flash"]
DEFAULT_METHODS = ["no_tool", "with_tool", "reflexion", "expel", "trace2skill", "xskill", "hive"]
METHOD_LABELS = {
    "no_tool": "CoT",
    "with_tool": "ReAct",
    "reflexion": "Reflexion",
    "expel": "ExpeL",
    "trace2skill": "Trace2Skill",
    "xskill": "XSkill",
    "hive": "HIVE",
}
METHOD_STYLES = {
    "no_tool": {"color": "#2563eb", "marker": "o", "linestyle": "-"},
    "with_tool": {"color": "#16a34a", "marker": "s", "linestyle": "-"},
    "reflexion": {"color": "#dc2626", "marker": "^", "linestyle": "-"},
    "expel": {"color": "#9333ea", "marker": "D", "linestyle": "-"},
    "trace2skill": {"color": "#f97316", "marker": "P", "linestyle": "-"},
    "xskill": {"color": "#0891b2", "marker": "X", "linestyle": "-"},
    "hive": {"color": "#111827", "marker": "*", "linestyle": "-"},
}
ROTATE_LABEL_180 = {"Agentic-MME", "VisBrowse-Bench"}


def _parse_csv(raw: str | None, default: list[str]) -> list[str]:
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _model_display_name(model: str) -> str:
    parts = []
    for part in model.split("-"):
        lowered = part.lower()
        if lowered.startswith("gpt"):
            parts.append("GPT" + part[3:])
        elif lowered.startswith("qwen"):
            parts.append("Qwen" + part[4:])
        elif lowered == "flash":
            parts.append("Flash")
        elif lowered == "chat":
            parts.append("Chat")
        elif lowered == "gpt":
            parts.append("GPT")
        elif lowered == "qwen":
            parts.append("Qwen")
        elif lowered in {"deepseek", "ds"}:
            parts.append("DeepSeek")
        elif lowered:
            parts.append(part)
    return "-".join(parts)


def _metric_files(method_root: Path, benchmark: str) -> list[Path]:
    return [
        method_root / benchmark / "test" / "metrics_at_k.json",
        method_root / benchmark / "output" / "test" / "metrics_at_k.json",
    ]


def _metric_file(method_root: Path, benchmark: str) -> Path | None:
    for path in _metric_files(method_root, benchmark):
        if path.exists():
            return path
    return None


def _method_has_any_metric(method_root: Path) -> bool:
    return any(_metric_file(method_root, benchmark) is not None for benchmark in BENCHMARKS)


def _available_methods(model_root: Path, requested_methods: list[str]) -> list[str]:
    methods = []
    for method in requested_methods:
        method_root = model_root / method
        if method_root.is_dir() and _method_has_any_metric(method_root):
            methods.append(method)
    return methods


def _available_models(output_root: Path, requested_models: list[str], requested_methods: list[str]) -> list[str]:
    models = []
    for model in requested_models:
        model_root = output_root / model
        if model_root.is_dir() and _available_methods(model_root, requested_methods):
            models.append(model)
    return models


def _load_metric(method_root: Path, metric_name: str, k: int = 4) -> list[float]:
    values = []
    for benchmark in BENCHMARKS:
        metrics_path = _metric_file(method_root, benchmark)
        if metrics_path is None:
            values.append(0.0)
            continue
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        values.append(float(metrics["k_metrics"][str(k)][metric_name]) * 100.0)
    return values


def _radar_angles(num_axes: int) -> np.ndarray:
    return np.linspace(0, 2 * np.pi, num_axes, endpoint=False)


def _radial_limit(panel_values: list[list[float]]) -> float:
    flat_values = [value for values in panel_values for value in values]
    upper = math.ceil((max(flat_values or [0.0]) + 5.0) / 10.0) * 10.0
    return min(100.0, max(20.0, upper))


def _set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 12,
            "axes.titlesize": 15,
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
        return rmax * 1.22
    if x > 0.55:
        return rmax * 1.14
    return rmax * 1.08


def _plot_benchmark_labels(ax, angles: np.ndarray, rmax: float) -> None:
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
            fontsize=10.5,
            fontweight="semibold",
            color="#1f2937",
            clip_on=False,
        )


def _plot_radar_panel(
    ax,
    *,
    model: str,
    metric_title: str,
    method_values: dict[str, list[float]],
) -> None:
    angles = _radar_angles(len(BENCHMARKS))
    closed_angles = np.concatenate([angles, angles[:1]])
    rmax = _radial_limit(list(method_values.values()))
    rticks = np.linspace(0, rmax, 5)[1:]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, rmax)
    _plot_benchmark_labels(ax, angles, rmax)

    for method, values in method_values.items():
        style = METHOD_STYLES.get(method, {"color": "#475569", "marker": "o", "linestyle": "-"})
        closed_values = np.array(values + values[:1])
        ax.plot(
            closed_angles,
            closed_values,
            linewidth=2.1,
            markersize=4.8,
            label=METHOD_LABELS.get(method, method),
            **style,
        )
        ax.fill(closed_angles, closed_values, color=style["color"], alpha=0.08)

    ax.set_yticks(rticks)
    ax.set_yticklabels([f"{tick:.0f}" for tick in rticks], color="#64748b", fontsize=8.5)
    ax.set_rlabel_position(90)
    ax.grid(color="#d9dee8", linewidth=0.85, alpha=0.76)
    ax.spines["polar"].set_color("#d1d5db")
    ax.spines["polar"].set_linewidth(0.9)
    ax.set_title(f"{_model_display_name(model)}\n{metric_title}", y=1.18, fontweight="bold", color="#111827")


def _load_model_metric_values(
    output_root: Path,
    model: str,
    methods: list[str],
    metric_name: str,
) -> dict[str, list[float]]:
    model_root = output_root / model
    values = {}
    for method in _available_methods(model_root, methods):
        values[method] = _load_metric(model_root / method, metric_name, k=4)
    return values


def _plot_model_grouped_radar(output_root: Path, models: list[str], methods: list[str], output_path: Path) -> None:
    _set_paper_style()
    models = _available_models(output_root, models, methods)
    if not models:
        raise FileNotFoundError(f"No matching metrics found under {output_root}")

    fig_width = max(8.0, 4.15 * len(models))
    fig, axes = plt.subplots(
        2,
        len(models),
        figsize=(fig_width, 8.9),
        subplot_kw={"projection": "polar"},
        squeeze=False,
    )

    for col, model in enumerate(models):
        average_values = _load_model_metric_values(output_root, model, methods, "average_at_k")
        pass_values = _load_model_metric_values(output_root, model, methods, "pass_at_k")
        _plot_radar_panel(
            axes[0][col],
            model=model,
            metric_title="Average@4 (%)",
            method_values=average_values,
        )
        _plot_radar_panel(
            axes[1][col],
            model=model,
            metric_title="Pass@4 (%)",
            method_values=pass_values,
        )

    handles_by_label = {}
    for ax in axes.ravel():
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            handles_by_label.setdefault(label, handle)
    fig.legend(
        handles_by_label.values(),
        handles_by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=max(1, len(handles_by_label)),
        frameon=False,
    )
    fig.subplots_adjust(left=0.045, right=0.955, top=0.9, bottom=0.095, wspace=0.34, hspace=0.58)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    args = parser.parse_args()

    output_path = args.output_dir / "baseline_average_pass_at_4_radar_by_model.pdf"
    _plot_model_grouped_radar(
        args.output_root,
        _parse_csv(args.models, DEFAULT_MODELS),
        _parse_csv(args.methods, DEFAULT_METHODS),
        output_path,
    )
    print(f"Saved {output_path}")
    print(f"Saved {output_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
