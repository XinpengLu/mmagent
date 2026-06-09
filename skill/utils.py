from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import yaml


_METADATA_PREFIXES = (
    "data_source",
    "task",
    "category",
    "domain",
    "subdomain",
    "subtask",
    "prompt_category",
    "eval_focus",
    "difficulty",
    "level",
)

_SKILL_GENERATION_METADATA_PREFIXES = (
    "data_source_",
    "domain_",
    "subdomain_",
    "category_",
    "prompt_category_",
    "eval_focus_",
    "difficulty_",
    "level_",
)

_GENERIC_CLUSTER_TASK_TAGS = {
    "general",
    "visual",
    "vision",
    "visual_reasoning",
    "image_analysis",
    "image-analysis",
    "image_processing",
    "image-processing",
    "tool_use",
    "tool",
    "validation",
    "visual_validation",
    "recovery",
    "recovery_from_failure",
    "error_handling",
}

DEFAULT_CALLABLE_TOOL_ORDER = ["code_interpreter", "web_search", "image_search", "visit", "zoom"]
DEFAULT_MEMORY_PROMPTS_PATH = "prompts/memory_rerank_prompts.yaml"
_configured_callable_tools = list(DEFAULT_CALLABLE_TOOL_ORDER)


def set_configured_callable_tools(raw: str) -> None:
    global _configured_callable_tools
    _configured_callable_tools = [name.strip() for name in raw.split(",") if name.strip()]


def callable_tools() -> list[str]:
    return _configured_callable_tools.copy()


def as_callable_tool(tool: Any) -> str:
    name = str(tool or "").strip()
    return name if name in _configured_callable_tools else ""


def tag_tokens(tags: Iterable[str]) -> set[str]:
    tokens = set()
    for tag in tags or []:
        tag_text = str(tag).strip().lower()
        if not tag_text:
            continue
        tokens.add(tag_text)
        tokens.update(part for part in re.split(r"[_\W]+", tag_text) if part)
    return tokens


def metadata_fields(tags: Iterable[str]) -> dict[str, set[str]]:
    fields: dict[str, set[str]] = {}
    for tag in tags or []:
        text = str(tag).strip().lower()
        if not text:
            continue
        for field in _METADATA_PREFIXES:
            prefix = f"{field}_"
            if text.startswith(prefix):
                fields.setdefault(field, set()).add(text[len(prefix):])
                break
    return fields


def record_metadata_fields(record: dict[str, Any]) -> dict[str, set[str]]:
    return metadata_fields(record.get("task_tags", []) or [])


def _metadata_field_overlap(
    query_fields: dict[str, set[str]],
    candidate_fields: dict[str, set[str]],
    field: str,
) -> set[str]:
    return query_fields.get(field, set()) & candidate_fields.get(field, set())


def metadata_field_conflicts(
    query_fields: dict[str, set[str]],
    candidate_fields: dict[str, set[str]],
    field: str,
) -> bool:
    query_values = query_fields.get(field)
    candidate_values = candidate_fields.get(field)
    return bool(query_values and candidate_values and not _metadata_field_overlap(query_fields, candidate_fields, field))


def metadata_field_matches(
    query_fields: dict[str, set[str]],
    candidate_fields: dict[str, set[str]],
    field: str,
) -> bool:
    return bool(_metadata_field_overlap(query_fields, candidate_fields, field))


def metadata_values(record: dict[str, Any], prefix: str) -> set[str]:
    tags = (str(tag).strip().lower() for tag in record.get("task_tags") or [])
    return {tag.removeprefix(prefix) for tag in tags if tag.startswith(prefix)}


def task_metadata_tags(record: dict[str, Any]) -> list[str]:
    tags = (str(tag).strip().lower() for tag in record.get("task_tags") or [])
    return [tag for tag in tags if tag.startswith(("data_source_", "task_"))]


def cluster_task_tags(record: dict[str, Any]) -> set[str]:
    tags = set()
    for raw_tag in record.get("task_tags", []) or []:
        tag = str(raw_tag).strip().lower()
        if (
            not tag
            or tag in _GENERIC_CLUSTER_TASK_TAGS
            or tag.startswith(_SKILL_GENERATION_METADATA_PREFIXES)
        ):
            continue
        tags.add(tag)
        if tag.startswith("task_"):
            tags.add(tag.removeprefix("task_"))
    return tags


def skill_name_token(tag: str) -> str:
    text = str(tag or "").strip().lower()
    ignored_prefixes = _SKILL_GENERATION_METADATA_PREFIXES + ("subtask_", "turncase_")
    if not text or text.startswith(ignored_prefixes):
        return ""
    text = text.removeprefix("task_")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return "" if text in {"general", "unknown", "reasoning"} else text


_MATCH_WEIGHTS = {
    "data_source": 1.25,
    "category": 1.25,
    "task": 1.25,
    "prompt_category": 1.20,
    "subtask": 1.20,
    "domain": 1.15,
    "subdomain": 1.15,
    "eval_focus": 1.15,
}

_MISMATCH_WEIGHTS = {
    "data_source": 0.35,
    "category": 0.25,
    "task": 0.25,
    "prompt_category": 0.25,
    "subtask": 0.25,
    "domain": 0.65,
    "subdomain": 0.65,
    "eval_focus": 0.65,
}


def metadata_selection_weight(candidate_tags: Iterable[str], query_metadata_tags: Iterable[str] | None) -> float:
    """Softly prefer skills from the same benchmark/task family."""
    query_fields = metadata_fields(query_metadata_tags or [])
    if not query_fields:
        return 1.0

    candidate_fields = metadata_fields(candidate_tags or [])
    if not candidate_fields:
        return 1.0

    weight = 1.0
    for field, query_values in query_fields.items():
        candidate_values = candidate_fields.get(field)
        if not candidate_values:
            continue
        if candidate_values & query_values:
            weight *= _MATCH_WEIGHTS.get(field, 1.05)
            if field == "task" and len(candidate_values) > 1:
                weight *= max(0.45, len(candidate_values) ** -0.5)
        elif field in _MISMATCH_WEIGHTS:
            weight *= _MISMATCH_WEIGHTS[field]
    return max(0.05, min(weight, 2.0))


def load_memory_prompt(prompt_key: str = "memory_rerank") -> str:
    """Load memory prompts from YAML so prompt text is not embedded in code."""
    prompt_path = Path(os.environ.get("MEMORY_PROMPTS_PATH", DEFAULT_MEMORY_PROMPTS_PATH))
    with prompt_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    prompt = data.get(prompt_key)
    if not isinstance(prompt, str) or not prompt.strip():
        raise KeyError(f"Prompt '{prompt_key}' not found in {prompt_path}")
    return prompt


def slugify(parts: Iterable[str], fallback: str = "general") -> str:
    text = "_".join(part for part in parts if part)
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text[:48] or fallback


def truncate_text(text: str, max_chars: int) -> str:
    text = text or ""
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    normalized = _normalize_rows(embeddings)
    return normalized @ normalized.T


def cosine_scores(query_embedding: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query = _normalize_rows(query_embedding.reshape(1, -1))
    return (query @ _normalize_rows(matrix).T)[0]
