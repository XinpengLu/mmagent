"""Shared schema helpers for structured experiences."""

from __future__ import annotations

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


def _as_rollout_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rollouts = []
    for item in value:
        if isinstance(item, dict):
            rollout = {str(k): v for k, v in item.items() if v is not None}
            if rollout:
                rollouts.append(rollout)
    return rollouts


def _normalize_action_policy(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    if isinstance(value, str) and value.strip():
        return {"instruction": value.strip()}
    return {}


def get_experience_content(value: Any) -> str:
    """Return the retrievable text content for either old or structured experience."""
    if isinstance(value, dict):
        for key in ("content", "experience", "text"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()
        nested = value.get("experience")
        if isinstance(nested, dict):
            return get_experience_content(nested)
        return ""
    return str(value).strip() if value is not None else ""


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
    failure_mode = data.get("failure_mode")
    if kind == "success" and failure_mode in ("", "none", "None"):
        failure_mode = None

    source_evidence = source.get("evidence_rollouts") if isinstance(source, dict) else None
    merged_source = {}
    if isinstance(data.get("source"), dict):
        merged_source.update(data["source"])
    if source:
        merged_source.update(source)

    stats = data.get("stats") if isinstance(data.get("stats"), dict) else {}
    record_id = exp_id or str(data.get("id", ""))
    confidence_default = 0.7 if kind == "success" else 0.6
    return {
        "id": record_id,
        "status": data.get("status", "active"),
        "kind": kind,
        "skill_ids": _as_list(data.get("skill_ids")),
        "task_tags": _as_list(data.get("task_tags")),
        "tool_tags": _as_list(data.get("tool_tags")),
        "visual_tags": _as_list(data.get("visual_tags")),
        "failure_mode": failure_mode if kind == "failure" else None,
        "confidence": _as_float(data.get("confidence"), confidence_default),
        "action_policy": _normalize_action_policy(data.get("action_policy")),
        "evidence_rollouts": _as_rollout_list(data.get("evidence_rollouts") or source_evidence),
        "content": content,
        "source": merged_source,
        "stats": {
            "support_count": int(stats.get("support_count", 1) or 1),
            "used_count": int(stats.get("used_count", 0) or 0),
            "success_after_used": int(stats.get("success_after_used", 0) or 0),
            "failure_after_used": int(stats.get("failure_after_used", 0) or 0),
        },
    }


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
            "visual_tags",
            "failure_mode",
            "confidence",
            "action_policy",
            "evidence_rollouts",
            "source",
            "stats",
        ):
            if key not in data and key in operation:
                data[key] = operation[key]
    else:
        data = {
            "kind": operation.get("kind") or operation.get("outcome") or default_kind,
            "skill_ids": operation.get("skill_ids", []),
            "task_tags": operation.get("task_tags", []),
            "tool_tags": operation.get("tool_tags", []),
            "visual_tags": operation.get("visual_tags", []),
            "failure_mode": operation.get("failure_mode"),
            "confidence": operation.get("confidence"),
            "action_policy": operation.get("action_policy", {}),
            "evidence_rollouts": operation.get("evidence_rollouts", []),
            "content": operation.get("content") or operation.get("experience") or operation.get("exp") or operation.get("text"),
            "source": operation.get("source", {}),
            "stats": operation.get("stats", {}),
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
    visual_tags = sorted({tag for r in normalized for tag in r.get("visual_tags", [])})
    skill_ids = sorted({tag for r in normalized for tag in r.get("skill_ids", [])})
    failure_modes = [r.get("failure_mode") for r in normalized if r.get("failure_mode")]
    support_count = sum(r.get("stats", {}).get("support_count", 1) for r in normalized)
    confidence_values = [r.get("confidence", 0.5) for r in normalized]
    evidence_rollouts = []
    for record in normalized:
        for rollout in record.get("evidence_rollouts", []):
            if rollout not in evidence_rollouts:
                evidence_rollouts.append(rollout)
    action_policy = {}
    for record in normalized:
        for key, value in record.get("action_policy", {}).items():
            action_policy.setdefault(key, value)
    source_ids = [r.get("id") for r in normalized if r.get("id")]

    return normalize_experience_record(
        exp_id,
        {
            "kind": kind,
            "skill_ids": skill_ids,
            "task_tags": task_tags,
            "tool_tags": tool_tags,
            "visual_tags": visual_tags,
            "failure_mode": failure_modes[0] if failure_modes else None,
            "confidence": sum(confidence_values) / len(confidence_values) if confidence_values else 0.5,
            "action_policy": action_policy,
            "evidence_rollouts": evidence_rollouts,
            "content": content,
            "source": {"merged_from": source_ids},
            "stats": {"support_count": support_count},
        },
    )


def format_experience_for_prompt(exp_id: str, value: Any) -> str:
    record = normalize_experience_record(exp_id, value)
    meta = [record["kind"]]
    if record["task_tags"]:
        meta.append("task_tags=" + ",".join(record["task_tags"]))
    if record["tool_tags"]:
        meta.append("tool_tags=" + ",".join(record["tool_tags"]))
    if record["visual_tags"]:
        meta.append("visual_tags=" + ",".join(record["visual_tags"]))
    if record["skill_ids"]:
        meta.append("skill_ids=" + ",".join(record["skill_ids"]))
    if record["failure_mode"]:
        meta.append("failure_mode=" + str(record["failure_mode"]))
    if record["action_policy"]:
        meta.append("action_policy=" + ";".join(f"{k}:{v}" for k, v in record["action_policy"].items()))
    return f"[{'; '.join(meta)}; confidence={record['confidence']:.2f}] {record['content']}"
