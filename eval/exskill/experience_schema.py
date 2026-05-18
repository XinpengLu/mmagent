"""Shared schema helpers for structured experiences."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional


ExperienceRecord = Dict[str, Any]
ExperienceLibrary = Dict[str, ExperienceRecord]

VALID_KINDS = {"success", "failure"}


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, tuple) or isinstance(value, set):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        if not value.strip():
            return []
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(value).strip()] if str(value).strip() else []


def _normalize_kind(value: Any, default: str = "success") -> str:
    kind = str(value or default).strip().lower()
    if kind in {"fail", "failed", "error", "negative"}:
        kind = "failure"
    if kind in {"correct", "positive"}:
        kind = "success"
    return kind if kind in VALID_KINDS else default


def _as_float(value: Any, default: float = 0.5) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, score))


def _clean_content_text(text: str) -> str:
    """Keep retrievable experience text free of duplicated structured metadata."""
    text = str(text or "").strip()
    if not text:
        return ""
    cleaned = re.sub(
        r"^\s*\[\s*(?:success|failure)\s*;[^\]]+\]\s*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    return cleaned or text


def get_experience_content(value: Any) -> str:
    """Return the retrievable text content for either old or structured experience."""
    if isinstance(value, dict):
        for key in ("content", "experience", "text"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return _clean_content_text(text)
        nested = value.get("experience")
        if isinstance(nested, dict):
            return get_experience_content(nested)
        return ""
    return _clean_content_text(str(value)) if value is not None else ""


def get_experience_kind(value: Any, default: str = "success") -> str:
    if isinstance(value, dict):
        return _normalize_kind(value.get("kind") or value.get("outcome"), default=default)
    return default


def normalize_experience_record(
    exp_id: str,
    value: Any,
    *,
    default_kind: str = "success",
    source: Optional[Dict[str, Any]] = None,
) -> ExperienceRecord:
    """Normalize legacy strings and partial dicts into the structured schema."""
    data = deepcopy(value) if isinstance(value, dict) else {}
    nested = data.get("experience")
    if isinstance(nested, dict):
        nested_data = deepcopy(nested)
        nested_data.update({k: v for k, v in data.items() if k not in {"experience"}})
        data = nested_data

    content = get_experience_content(data if data else value)
    kind = get_experience_kind(data, default=default_kind)

    stats = data.get("stats") if isinstance(data.get("stats"), dict) else {}
    record_id = exp_id or str(data.get("id", ""))
    confidence_default = 0.7 if kind == "success" else 0.6
    record = {
        "id": record_id,
        "kind": kind,
        "skill_ids": _as_list(data.get("skill_ids")),
        "task_tags": _as_list(data.get("task_tags")),
        "tool_tags": _as_list(data.get("tool_tags")),
        "confidence": _as_float(data.get("confidence"), confidence_default),
        "content": content,
        "stats": {
            "support_count": int(stats.get("support_count", 1) or 1),
            "used_count": int(stats.get("used_count", 0) or 0),
            "success_after_used": int(stats.get("success_after_used", 0) or 0),
            "failure_after_used": int(stats.get("failure_after_used", 0) or 0),
        },
    }
    for optional_key in ("code_template", "anti_pattern"):
        val = data.get(optional_key)
        if isinstance(val, str) and val.strip():
            record[optional_key] = val.strip()
    return record


def normalize_experience_library(raw: Any) -> ExperienceLibrary:
    """Load either {experiences:{...}} or direct {E0: ...} into structured records."""
    if isinstance(raw, dict) and "experiences" in raw and isinstance(raw["experiences"], dict):
        raw = raw["experiences"]
    if not isinstance(raw, dict):
        return {}

    normalized: ExperienceLibrary = {}
    for exp_id, value in raw.items():
        content = get_experience_content(value)
        if not content:
            continue
        normalized[str(exp_id)] = normalize_experience_record(str(exp_id), value)
    return normalized


def compact_experience_record(value: Any, exp_id: str = "") -> ExperienceRecord:
    """Serialize the minimal experience schema used by retrieval/injection/skill clustering."""
    record = normalize_experience_record(exp_id, value)
    compact: ExperienceRecord = {
        "kind": record["kind"],
        "content": record["content"],
    }
    for key in ("task_tags", "tool_tags", "skill_ids"):
        if record.get(key):
            compact[key] = record[key]
    for key in ("code_template", "anti_pattern"):
        if record.get(key):
            compact[key] = record[key]
    return compact


def compact_experience_operation(operation: Dict[str, Any]) -> Dict[str, Any]:
    """Compact an add/modify operation for debug files."""
    if not isinstance(operation, dict):
        return operation
    compact = {"option": operation.get("option", "add")}
    if operation.get("modified_from"):
        compact["modified_from"] = str(operation["modified_from"])
    compact["experience"] = compact_experience_record(operation.get("experience", operation))
    return compact


def normalize_experience_operation(
    operation: Dict[str, Any],
    *,
    default_kind: str = "success",
    source: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize LLM-produced add/modify operations while preserving metadata."""
    if not isinstance(operation, dict):
        return None

    option = str(operation.get("option", "add")).strip().lower()
    if option not in {"add", "modify"}:
        option = "add"

    payload = operation.get("experience")
    if isinstance(payload, dict):
        data = deepcopy(payload)
        for key in (
            "kind",
            "outcome",
            "skill_ids",
            "task_tags",
            "tool_tags",
            "confidence",
            "stats",
            "code_template",
            "anti_pattern",
        ):
            if key not in data and key in operation:
                data[key] = operation[key]
    else:
        data = {
            "kind": operation.get("kind") or operation.get("outcome") or default_kind,
            "skill_ids": operation.get("skill_ids", []),
            "task_tags": operation.get("task_tags", []),
            "tool_tags": operation.get("tool_tags", []),
            "confidence": operation.get("confidence"),
            "content": operation.get("content") or operation.get("experience") or operation.get("exp") or operation.get("text"),
            "stats": operation.get("stats", {}),
            "code_template": operation.get("code_template"),
            "anti_pattern": operation.get("anti_pattern"),
        }

    record = normalize_experience_record("", data, default_kind=default_kind, source=source)
    if not record["content"]:
        return None

    normalized = {"option": option, "experience": record}
    if option == "modify" and operation.get("modified_from"):
        normalized["modified_from"] = str(operation["modified_from"])
    return normalized


def make_record_for_id(exp_id: str, operation: Any) -> ExperienceRecord:
    payload = operation.get("experience", operation) if isinstance(operation, dict) else operation
    return normalize_experience_record(exp_id, payload)


def reassign_experience_ids(experiences: Dict[str, Any]) -> ExperienceLibrary:
    """Renumber experiences as E0, E1, ... and update each record's id field."""
    final: ExperienceLibrary = {}

    def sort_key(item):
        key = item[0]
        return int(key[1:]) if isinstance(key, str) and key.startswith("E") and key[1:].isdigit() else 10**9

    for idx, (_, value) in enumerate(sorted(experiences.items(), key=sort_key)):
        exp_id = f"E{idx}"
        final[exp_id] = normalize_experience_record(exp_id, value)
    return final


def merge_record_metadata(exp_id: str, records: Iterable[Any], content: str) -> ExperienceRecord:
    """Create one structured record from several same-kind records and merged content."""
    normalized = [normalize_experience_record("", r) for r in records if get_experience_content(r)]
    if not normalized:
        return normalize_experience_record(exp_id, {"content": content})

    kinds = {r["kind"] for r in normalized}
    kind = normalized[0]["kind"] if len(kinds) == 1 else "failure"
    task_tags = sorted({tag for r in normalized for tag in r.get("task_tags", [])})
    tool_tags = sorted({tag for r in normalized for tag in r.get("tool_tags", [])})
    skill_ids = sorted({tag for r in normalized for tag in r.get("skill_ids", [])})
    support_count = sum(r.get("stats", {}).get("support_count", 1) for r in normalized)
    used_count = sum(r.get("stats", {}).get("used_count", 0) for r in normalized)
    success_after_used = sum(r.get("stats", {}).get("success_after_used", 0) for r in normalized)
    failure_after_used = sum(r.get("stats", {}).get("failure_after_used", 0) for r in normalized)
    confidence_values = [r.get("confidence", 0.5) for r in normalized]

    return normalize_experience_record(
        exp_id,
        {
            "kind": kind,
            "skill_ids": skill_ids,
            "task_tags": task_tags,
            "tool_tags": tool_tags,
            "confidence": sum(confidence_values) / len(confidence_values) if confidence_values else 0.5,
            "content": content,
            "stats": {
                "support_count": support_count,
                "used_count": used_count,
                "success_after_used": success_after_used,
                "failure_after_used": failure_after_used,
            },
        },
    )


def format_experience_for_prompt(exp_id: str, value: Any) -> str:
    record = normalize_experience_record(exp_id, value)
    meta = [record["kind"]]
    if record["task_tags"]:
        meta.append("task=" + ",".join(record["task_tags"]))
    if record["tool_tags"]:
        meta.append("tool=" + ",".join(record["tool_tags"]))
    parts = [f"[{'; '.join(meta)}] {record['content']}"]
    if record.get("anti_pattern"):
        parts.append(f"  Avoid: {record['anti_pattern']}")
    if record.get("code_template"):
        parts.append(f"  Code: {record['code_template']}")
    return "\n".join(parts)
