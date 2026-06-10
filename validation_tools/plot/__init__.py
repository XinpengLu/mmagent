"""Publication-style plotting entry points for validation analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .memory_artifact_plots import _build_memory_artifact_plots
from .summary_plots import _plot_efficiency, _plot_self_evolution_overview


def build_all_plots(
    figures_dir: Path,
    rows: List[Dict[str, Any]],
    run_root: Path | None = None,
) -> List[Path]:
    """Generate all report figures and return the written PDF paths."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in figures_dir.glob("*"):
        if stale_file.is_file():
            stale_file.unlink()
    written = [
        _plot_self_evolution_overview(figures_dir / "self_evolution_overview.pdf", rows),
        _plot_efficiency(figures_dir / "efficiency_tokens_runtime.pdf", rows),
    ]
    if run_root is not None:
        written.extend(_build_memory_artifact_plots(figures_dir, run_root))
    return [path for path in written if path is not None]
