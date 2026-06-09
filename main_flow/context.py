"""Shared configuration helpers for benchmark main flow."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from configs.runtime import RuntimeConfig


DEFAULT_PROMPT_TOOLS = ("multi_tool_agent", "web_search, visit, code_interpreter")
SEARCH_PROMPT_TOOLS = ("multi_tool_agent_search", "web_search, image_search, visit, code_interpreter")

BENCHMARK_PROMPT_TOOLS = {
    "TIR-Bench": ("multi_tool_agent_code", "code_interpreter"),
    "AgentVista": SEARCH_PROMPT_TOOLS,
    "MMSearch-Plus": SEARCH_PROMPT_TOOLS,
    "MMBrowseComp": SEARCH_PROMPT_TOOLS,
}


@dataclass(frozen=True)
class MainFlowContext:
    data_root: Path
    bench_root: Path
    split_root: Path
    output_root: Path
    log_root: Path
    memory_root_base: Path
    system_prompts: Path
    tool_config: Path
    settings: dict[str, Any] = field(default_factory=dict)


def _count_json_items(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return sum(1 for line in f if line.strip())
        return len(json.load(f))


def _default_prompt_and_tools(benchmark: str) -> tuple[str, str]:
    return BENCHMARK_PROMPT_TOOLS.get(benchmark, DEFAULT_PROMPT_TOOLS)


def build_context(runtime: "RuntimeConfig") -> MainFlowContext:
    paths = runtime.paths
    return MainFlowContext(
        data_root=paths.data_root,
        bench_root=paths.bench_root,
        split_root=paths.split_root,
        output_root=paths.output_root,
        log_root=paths.log_root,
        memory_root_base=paths.memory_root_base,
        system_prompts=paths.system_prompts,
        tool_config=paths.tool_config,
        settings=runtime.settings,
    )
