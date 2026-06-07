"""Shared configuration helpers for benchmark main flow."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROMPT_TOOLS = ("multi_tool_agent", "web_search, visit, code_interpreter")
SEARCH_PROMPT_TOOLS = ("multi_tool_agent_search", "web_search, image_search, visit, code_interpreter")

BENCHMARK_PROMPT_TOOLS = {
    "TIR-Bench": ("multi_tool_agent_code", "code_interpreter"),
    "AgentVista": SEARCH_PROMPT_TOOLS,
    "MMSearch-Plus": SEARCH_PROMPT_TOOLS,
    "MMBrowseComp": SEARCH_PROMPT_TOOLS,
}


def _project_root() -> Path:
    """Return the mmagent source root regardless of the caller's cwd."""
    return Path(__file__).resolve().parents[1]


def _data_root(default: Path | None = None) -> Path:
    """Return DATA_ROOT or the mmagent-local data root used by scripts."""
    root = _project_root()
    fallback = default or root
    return Path(os.environ.get("DATA_ROOT", fallback)).resolve()


@dataclass(frozen=True)
class MainFlowContext:
    data_root: Path
    bench_root: Path
    split_root: Path
    output_root: Path
    log_root: Path
    benchmark_count: int


def _count_json_items(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return len(json.load(f))


def _default_prompt_and_tools(benchmark: str) -> tuple[str, str]:
    return BENCHMARK_PROMPT_TOOLS.get(benchmark, DEFAULT_PROMPT_TOOLS)


def build_context(benchmark_count: int) -> MainFlowContext:
    resolved_data_root = _data_root(_project_root())
    return MainFlowContext(
        data_root=resolved_data_root,
        bench_root=Path(os.environ.get("BENCH_ROOT") or resolved_data_root / "benchmark" / "processed"),
        split_root=Path(os.environ.get("BENCHMARK_SPLIT_ROOT") or resolved_data_root / "benchmark" / "split"),
        output_root=Path(os.environ.get("OUTPUT_ROOT") or resolved_data_root / "output" / "full_benchmark"),
        log_root=Path(os.environ.get("LOG_ROOT") or resolved_data_root / "output" / "full_benchmark" / "_logs"),
        benchmark_count=benchmark_count,
    )
