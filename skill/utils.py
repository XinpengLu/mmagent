"""Utility helpers for modular skill memory."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml
from utils.env import env_float


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

CALLABLE_TOOL_NAMES = {"code_interpreter", "web_search", "visit", "image_search", "zoom"}
DEFAULT_CALLABLE_TOOL_ORDER = ["code_interpreter", "web_search", "image_search", "visit", "zoom"]
_configured_callable_tools = list(DEFAULT_CALLABLE_TOOL_ORDER)


def set_configured_callable_tools(raw: Optional[str]) -> None:
    """Set phase-local callable tools from the benchmark prompt/tool mapping."""
    global _configured_callable_tools
    tools = [
        tool.strip()
        for tool in str(raw or "").split(",")
        if tool.strip() in CALLABLE_TOOL_NAMES
    ]
    _configured_callable_tools = tools or list(DEFAULT_CALLABLE_TOOL_ORDER)


def configured_callable_tools(raw: Optional[str] = None) -> List[str]:
    """Return enabled callable tools in prompt order."""
    if raw is None:
        return list(_configured_callable_tools)
    value = raw
    tools = [
        tool.strip()
        for tool in str(value or "").split(",")
        if tool.strip() in CALLABLE_TOOL_NAMES
    ]
    return tools or list(DEFAULT_CALLABLE_TOOL_ORDER)


def enabled_callable_tools(raw: Optional[str] = None) -> set:
    return set(configured_callable_tools(raw))


def as_callable_tool(tool: Any, raw: Optional[str] = None) -> str:
    name = str(tool or "").strip()
    return name if name in enabled_callable_tools(raw) else ""


def tag_tokens(tags: Iterable[str]) -> set:
    tokens = set()
    for tag in tags or []:
        tag_text = str(tag).strip().lower()
        if not tag_text:
            continue
        tokens.add(tag_text)
        tokens.update(part for part in re.split(r"[_\W]+", tag_text) if part)
    return tokens


def metadata_fields(tags: Iterable[str]) -> Dict[str, set]:
    fields: Dict[str, set] = {}
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


def record_metadata_fields(record: Dict[str, Any]) -> Dict[str, set]:
    return metadata_fields(record.get("task_tags", []) or [])


def metadata_field_conflicts(query_fields: Dict[str, set], candidate_fields: Dict[str, set], field: str) -> bool:
    query_values = query_fields.get(field)
    candidate_values = candidate_fields.get(field)
    return bool(query_values and candidate_values and not (query_values & candidate_values))


def metadata_field_matches(query_fields: Dict[str, set], candidate_fields: Dict[str, set], field: str) -> bool:
    query_values = query_fields.get(field)
    candidate_values = candidate_fields.get(field)
    return bool(query_values and candidate_values and (query_values & candidate_values))


def metadata_values(record: Dict[str, Any], prefix: str) -> set:
    values = set()
    for tag in record.get("task_tags") or []:
        text = str(tag).strip().lower()
        if text.startswith(prefix):
            values.add(text[len(prefix):])
    return values


def task_metadata_tags(record: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    for tag in record.get("task_tags") or []:
        text = str(tag).strip().lower()
        if text.startswith("data_source_") or text.startswith("task_"):
            tags.append(text)
    return tags


def cluster_task_tags(record: Dict[str, Any]) -> set:
    tags = set()
    for tag in record.get("task_tags", []) or []:
        tag = str(tag).strip().lower()
        if not tag or tag in _GENERIC_CLUSTER_TASK_TAGS:
            continue
        if any(tag.startswith(prefix) for prefix in _SKILL_GENERATION_METADATA_PREFIXES):
            continue
        if tag.startswith("task_"):
            tags.add(tag)
            tags.add(tag[len("task_"):])
        else:
            tags.add(tag)
    return tags


def skill_name_token(tag: str) -> str:
    text = str(tag or "").strip().lower()
    if not text:
        return ""
    for prefix in _SKILL_GENERATION_METADATA_PREFIXES + ("data_source_", "subtask_", "turncase_"):
        if text.startswith(prefix):
            return ""
    if text.startswith("task_"):
        text = text[len("task_"):]
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if text in {"general", "unknown", "reasoning"}:
        return ""
    return text


def metadata_selection_weight(candidate_tags: Iterable[str], query_metadata_tags: Optional[Iterable[str]]) -> float:
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
            if field in {"data_source", "category", "task"}:
                weight *= 1.25
                if field == "task" and len(candidate_values) > 1:
                    weight *= max(0.45, 1.0 / (len(candidate_values) ** 0.5))
            elif field in {"prompt_category", "subtask"}:
                weight *= 1.20
            elif field in {"domain", "subdomain", "subtask", "prompt_category", "eval_focus"}:
                weight *= 1.15
            else:
                weight *= 1.05
        else:
            if field == "data_source":
                weight *= env_float("MEMORY_METADATA_DATASOURCE_MISMATCH_WEIGHT", 0.35)
            elif field in {"category", "task", "prompt_category", "subtask"}:
                weight *= env_float("MEMORY_METADATA_CATEGORY_MISMATCH_WEIGHT", 0.25)
            elif field in {"domain", "subdomain", "subtask", "prompt_category", "eval_focus"}:
                weight *= env_float("MEMORY_METADATA_DOMAIN_MISMATCH_WEIGHT", 0.65)
    return max(0.05, min(weight, 2.0))


def load_memory_prompt(prompt_key: str = "memory_rerank") -> str:
    """Load memory prompts from YAML so prompt text is not embedded in code."""
    prompt_path = Path(
        os.environ.get("MEMORY_PROMPTS_PATH", "prompts/memory_rerank_prompts.yaml")
    )
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


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms
    return normalized @ normalized.T


def cosine_scores(query_embedding: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query_embedding.reshape(1, -1)
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    m_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1.0
    m_norm[m_norm == 0] = 1.0
    return ((q / q_norm) @ (matrix / m_norm).T)[0]
