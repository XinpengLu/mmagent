"""Publication-style plotting entry points for validation analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .memory_artifact_plots import _build_memory_artifact_plots
from .summary_plots import _plot_diagnostics, _plot_efficiency, _plot_memory, _plot_performance


def build_all_plots(
    figures_dir: Path,
    rows: List[Dict[str, Any]],
    run_root: Path | None = None,
) -> List[Path]:
    """Generate all report figures and return the written PDF paths."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    written = [
        _plot_performance(figures_dir / "performance_curve.pdf", rows),
        _plot_efficiency(figures_dir / "efficiency_tokens_runtime.pdf", rows),
        _plot_memory(figures_dir / "memory_and_skill_usage.pdf", rows),
        _plot_diagnostics(figures_dir / "validation_diagnostics.pdf", rows),
    ]
    if run_root is not None:
        written.extend(_build_memory_artifact_plots(figures_dir, run_root))
    return [path for path in written if path is not None]
