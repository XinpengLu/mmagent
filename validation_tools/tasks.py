"""Task-name inference helpers for validation metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from utils.artifacts import MEMORY_USAGE_FILENAME, SAMPLE_SUMMARY_FILENAME, read_json


_METADATA_TASK_FIELDS = ("task", "domain", "category", "prompt_category", "subtask")


def task_of_sample(sample: Dict[str, Any]) -> str:
    for field in _METADATA_TASK_FIELDS:
        value = str(sample.get(field) or "").strip()
        if value:
            return value
    return _task_from_id(str(sample.get("doc_id") or sample.get("question_id") or ""))


def _task_from_id(identifier: str) -> str:
    text = str(identifier or "").strip()
    if not text:
        return "unknown"
    parts = text.split("_")
    if len(parts) == 3 and parts[0].isalpha() and parts[-1].isdigit():
        return parts[1].lower()
    return "unknown"


def _task_from_metadata_tags(tags: Iterable[str]) -> str:
    for tag in tags or []:
        text = str(tag or "").strip().lower()
        if text.startswith("task_") and len(text) > 5:
            return text[5:]
        if text.startswith("domain_") and len(text) > 7:
            return text[7:]
        if text.startswith("category_") and len(text) > 9:
            return text[9:]
        if text.startswith("subdomain_") and len(text) > 10:
            return text[10:]
    return "unknown"


def task_from_sample_dir(sample_dir: Path) -> str:
    metrics_path = sample_dir / SAMPLE_SUMMARY_FILENAME
    if metrics_path.exists():
        metrics = read_json(metrics_path)
        for key in ("task", "domain", "category", "subdomain", "prompt_category"):
            if metrics.get(key):
                return str(metrics[key])
        identifier = metrics.get("question_id") or metrics.get("doc_id") or sample_dir.name
        inferred = _task_from_id(str(identifier))
        if inferred != "unknown":
            return inferred

    usage_path = sample_dir / MEMORY_USAGE_FILENAME
    if usage_path.exists():
        usage = read_json(usage_path)
        info = usage.get("skill_retrieval_info") or {}
        inferred = _task_from_metadata_tags(info.get("query_metadata_tags") or [])
        if inferred != "unknown":
            return inferred
        context = str(usage.get("rerank_metadata_context") or "")
        tags: List[str] = []
        if ":" in context:
            tags = [part.strip() for part in context.split(":", 1)[1].split(",")]
        inferred = _task_from_metadata_tags(tags)
        if inferred != "unknown":
            return inferred

    return _task_from_id(sample_dir.name)
