"""Shared artifact filenames and lightweight JSON helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


RESULTS_FILENAME = "results.jsonl"
TRAJECTORY_FILENAME = "traj.jsonl"
ROLLOUT_METRICS_FILENAME = "metrics.json"
INJECTED_SYSTEM_PROMPT_FILENAME = "injected_system_prompt.txt"
SAMPLE_SUMMARY_FILENAME = "metrics_sample.json"
DATASET_SUMMARY_FILENAME = "metrics_at_k.json"
AVERAGE_PASS_FILENAME = "average_pass_at_k.json"
MEMORY_USAGE_FILENAME = "tt_memory_usage.json"
RETRIEVAL_INFO_FILENAME = "tt_decomposition_info.txt"


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, *, indent: int = 4) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def save_trajectory(save_directory: str | Path, item: Any) -> None:
    """Append one trajectory record to traj.jsonl."""
    target = Path(save_directory) / TRAJECTORY_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def clean_rollout_artifacts(save_directory: str | Path) -> None:
    """Remove files produced by one failed rollout attempt."""
    target = Path(save_directory)
    for filename in (TRAJECTORY_FILENAME, ROLLOUT_METRICS_FILENAME, INJECTED_SYSTEM_PROMPT_FILENAME):
        path = target / filename
        if path.exists():
            path.unlink()

    for path in target.glob("*"):
        if path.name.startswith(("tool_image_", "observation_")) and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            path.unlink()


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return
    with source.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{source}:{line_no} must contain a JSON object.")
            yield item
