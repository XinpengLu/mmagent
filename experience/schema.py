from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional


ExperienceRecord = Dict[str, Any]
ExperienceLibrary = Dict[str, ExperienceRecord]

OPTIONAL_TEXT_FIELDS = (
    "code_template",
    "anti_pattern",
    "recovery_status",
    "source_validation_stage",
    "source_validation_sample_id",
)
OPTIONAL_LIST_FIELDS = (
    "source_failure_ids",
    "source_success_ids",
    "recovery_source_ids",
)
OPERATION_METADATA_FIELDS = (
    "kind",
    "skill_ids",
    "task_tags",
    "tool_tags",
    "stats",
    "tool_sequence",
    "validation_checks",
    "fallback_plan",
    "code_template",
    "anti_pattern",
    "recovery_status",
    "source_failure_ids",
    "source_success_ids",
    "recovery_source_ids",
    "source_validation_stage",
    "source_validation_sample_id",
)


def text_list(value: Any, max_items: Optional[int] = None, *, unique: bool = False) -> List[str]:
    values = value if isinstance(value, list) else ([value] if value else [])
    items: List[str] = []
    for item in values[:max_items]:
        text = str(item or "").strip()
        if text and (not unique or text not in items):
            items.append(text)
    return items


def normalize_tool_sequence(value: Any, max_items: int = 5) -> List[Dict[str, str]]:
    """Normalize an ordered tool plan into [{tool, purpose, expected_output}]."""
    return [
        {
            "tool": str(item.get("tool", "")).strip(),
            "purpose": str(item.get("purpose", "")).strip(),
            "expected_output": str(item.get("expected_output", "")).strip(),
        }
        for item in (value or [])[:max_items]
    ]


def normalize_experience_record(
    exp_id: str,
    value: Any,
    *,
    source: Optional[Dict[str, Any]] = None,
) -> ExperienceRecord:
    """Coerce one generated or constructed record into the canonical shape."""
    data = deepcopy(value) if isinstance(value, dict) else {"content": str(value or "")}

    content = str(data.get("content", "")).strip()
    kind = str(data.get("kind") or "success").strip().lower()
    if kind not in {"success", "failure"}:
        kind = "success"

    stats = data.get("stats") or {}
    record_id = exp_id or str(data.get("id", ""))
    normalized_stats = {
        "support_count": int(stats.get("support_count", 1) or 1),
        "used_count": int(stats.get("used_count", 0) or 0),
        "success_after_used": int(stats.get("success_after_used", 0) or 0),
        "failure_after_used": int(stats.get("failure_after_used", 0) or 0),
    }

    for key, value in stats.items():
        if key in normalized_stats:
            continue
        if isinstance(value, (int, float, str, bool)):
            normalized_stats[key] = value

    record = {
        "id": record_id,
        "kind": kind,
        "skill_ids": text_list(data.get("skill_ids")),
        "task_tags": text_list(data.get("task_tags")),
        "tool_tags": text_list(data.get("tool_tags")),
        "content": content,
        "stats": normalized_stats,
    }
    tool_sequence = normalize_tool_sequence(data.get("tool_sequence"))
    
    if tool_sequence:
        record["tool_sequence"] = tool_sequence
    validation_checks = text_list(data.get("validation_checks"), 6, unique=True)
    
    if validation_checks:
        record["validation_checks"] = validation_checks
    fallback_plan = text_list(data.get("fallback_plan"), 6, unique=True)
    
    if fallback_plan:
        record["fallback_plan"] = fallback_plan
    evidence_rollouts = (data.get("evidence_rollouts") or (source or {}).get("evidence_rollouts") or [])[:8]
    
    if evidence_rollouts:
        record["evidence_rollouts"] = evidence_rollouts

    for optional_key in OPTIONAL_TEXT_FIELDS:
        val = data.get(optional_key)
        if isinstance(val, str) and val.strip():
            record[optional_key] = val.strip()
    
    for optional_key in OPTIONAL_LIST_FIELDS:
        vals = text_list(data.get(optional_key))
        if vals:
            record[optional_key] = vals
    return record


def normalize_experience_library(raw: Any) -> ExperienceLibrary:
    """Validate and normalize an in-memory runtime mapping."""
    if not raw:
        return {}

    return {
        str(exp_id): normalize_experience_record(str(exp_id), value)
        for exp_id, value in raw.items()
    }


def _serialize_experience_library(experiences: Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical JSON payload written to disk."""
    normalized = normalize_experience_library(experiences)
    records = {
        exp_id: normalize_experience_record(exp_id, record)
        for exp_id, record in normalized.items()
    }
    return {
        "experiences": records,
    }


def _normalize_experience_operation(
    operation: Dict[str, Any],
    *,
    source: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize LLM-produced add/modify operations while preserving metadata."""
    if not isinstance(operation, dict):
        return None

    option = str(operation.get("option", "add")).strip().lower()
    if option not in {"add", "modify"}:
        option = "add"

    data = {key: operation.get(key) for key in OPERATION_METADATA_FIELDS}
    data["content"] = operation.get("content")

    record = normalize_experience_record("", data, source=source)
    if not record["content"]:
        return None

    normalized = {"option": option, "experience": record}
    if option == "modify" and operation.get("modified_from"):
        normalized["modified_from"] = str(operation["modified_from"])
    return normalized


def _merge_sorted_record_list(records: List[ExperienceRecord], key: str) -> List[str]:
    return sorted({tag for record in records for tag in record.get(key, [])})


def _append_unique_until_limit(
    target: List[Any],
    values: Iterable[Any],
    limit: int,
) -> None:
    for value in values or []:
        if value not in target:
            target.append(value)
        if len(target) >= limit:
            break


def _append_unique_texts(target: List[str], values: Iterable[Any]) -> None:
    for value in values or []:
        if value and value not in target:
            target.append(value)


def _merge_record_sequences(records: List[ExperienceRecord]) -> Dict[str, Any]:
    merged = {
        "tool_sequence": [],
        "validation_checks": [],
        "fallback_plan": [],
        "evidence_rollouts": [],
        "code_template": "",
        "anti_pattern": "",
    }
    for record in records:
        _append_unique_until_limit(merged["tool_sequence"], record.get("tool_sequence", []), 5)
        _append_unique_until_limit(merged["validation_checks"], record.get("validation_checks", []), 6)
        _append_unique_until_limit(merged["fallback_plan"], record.get("fallback_plan", []), 4)
        _append_unique_until_limit(merged["evidence_rollouts"], record.get("evidence_rollouts", []), 8)
        if not merged["code_template"] and record.get("code_template"):
            merged["code_template"] = record["code_template"]
        if not merged["anti_pattern"] and record.get("anti_pattern"):
            merged["anti_pattern"] = record["anti_pattern"]
    return merged


def _merge_validation_and_recovery(records: List[ExperienceRecord]) -> Dict[str, Any]:
    merged = {
        "validation_used": 0,
        "validation_score_sum": 0.0,
        "validation_pass_sum": 0.0,
        "last_validation_stage": "",
        "recovered_from_failure": False,
        "recovery_statuses": [],
        "source_failure_ids": [],
        "source_success_ids": [],
        "recovery_source_ids": [],
    }
    for record in records:
        stats = record["stats"]
        count = int(stats.get("validation_used_count", 0) or 0)
        merged["validation_used"] += count
        merged["validation_score_sum"] += float(
            stats.get("validation_score_sum", float(stats.get("validation_average_score", 0.0) or 0.0) * count)
            or 0.0
        )
        merged["validation_pass_sum"] += float(
            stats.get("validation_pass_sum", float(stats.get("validation_pass_rate", 0.0) or 0.0) * count)
            or 0.0
        )
        stage = str(stats.get("last_validation_stage") or "")
        if stage:
            merged["last_validation_stage"] = stage
        status = str(record.get("recovery_status") or "").strip()
        if status:
            merged["recovery_statuses"].append(status)
        merged["recovered_from_failure"] = (
            merged["recovered_from_failure"] or bool(stats.get("recovered_from_failure")) or bool(status)
        )
        _append_unique_texts(merged["source_failure_ids"], record.get("source_failure_ids", []))
        _append_unique_texts(merged["source_success_ids"], record.get("source_success_ids", []))
        _append_unique_texts(merged["recovery_source_ids"], record.get("recovery_source_ids", []))
    return merged


def _merged_recovery_status(statuses: List[str]) -> str:
    if "rejected" in statuses:
        return "rejected"
    if "candidate_unverified" in statuses:
        return "candidate_unverified"
    if statuses and all(status == "validated" for status in statuses):
        return "validated"
    return ""


def _merge_record_stats(records: List[ExperienceRecord], recovery: Dict[str, Any]) -> Dict[str, Any]:
    stats = [record["stats"] for record in records]
    merged_stats = {
        "support_count": sum(item.get("support_count", 1) for item in stats),
        "used_count": sum(item.get("used_count", 0) for item in stats),
        "success_after_used": sum(item.get("success_after_used", 0) for item in stats),
        "failure_after_used": sum(item.get("failure_after_used", 0) for item in stats),
    }
    validation_used = recovery["validation_used"]
    if validation_used:
        merged_stats.update(
            {
                "validation_used_count": validation_used,
                "validation_score_sum": round(recovery["validation_score_sum"], 6),
                "validation_pass_sum": round(recovery["validation_pass_sum"], 6),
                "validation_average_score": round(recovery["validation_score_sum"] / validation_used, 6),
                "validation_pass_rate": round(recovery["validation_pass_sum"] / validation_used, 6),
            }
        )
        if recovery["last_validation_stage"]:
            merged_stats["last_validation_stage"] = recovery["last_validation_stage"]
    if recovery["recovered_from_failure"]:
        merged_stats["recovered_from_failure"] = True
    return merged_stats


def _merge_record_metadata(exp_id: str, records: Iterable[Any], content: str) -> ExperienceRecord:
    """Create one structured record from several same-kind records and merged content."""
    normalized = [normalize_experience_record("", record) for record in records if str(record.get("content", "")).strip()]
    if not normalized:
        return normalize_experience_record(exp_id, {"content": content})

    kinds = {record["kind"] for record in normalized}
    sequence_fields = _merge_record_sequences(normalized)
    recovery = _merge_validation_and_recovery(normalized)
    payload = {
        "kind": normalized[0]["kind"] if len(kinds) == 1 else "failure",
        "skill_ids": _merge_sorted_record_list(normalized, "skill_ids"),
        "task_tags": _merge_sorted_record_list(normalized, "task_tags"),
        "tool_tags": _merge_sorted_record_list(normalized, "tool_tags"),
        "content": content,
        "stats": _merge_record_stats(normalized, recovery),
        **sequence_fields,
    }
    status = _merged_recovery_status(recovery["recovery_statuses"])
    if recovery["recovered_from_failure"] and status:
        payload["recovery_status"] = status
    for key in ("source_failure_ids", "source_success_ids", "recovery_source_ids"):
        if recovery[key]:
            payload[key] = recovery[key]

    return normalize_experience_record(exp_id, payload)


def format_experience_for_prompt(exp_id: str, value: Any) -> str:
    record = normalize_experience_record(exp_id, value)
    meta = [record["kind"]]
    if record["task_tags"]:
        meta.append("task=" + ",".join(record["task_tags"]))
    if record["tool_tags"]:
        meta.append("tool=" + ",".join(record["tool_tags"]))
    parts = [f"[{'; '.join(meta)}] {record['content']}"]
    if record.get("tool_sequence"):
        steps = []
        for idx, step in enumerate(record["tool_sequence"], 1):
            expected = f" -> expect {step.get('expected_output')}" if step.get("expected_output") else ""
            steps.append(f"{idx}. {step.get('tool')}: {step.get('purpose')}{expected}")
        parts.append("  Tool sequence: " + " | ".join(steps))
    if record.get("validation_checks"):
        parts.append("  Validate: " + " | ".join(record["validation_checks"]))
    if record.get("fallback_plan"):
        parts.append("  Fallback: " + " | ".join(record["fallback_plan"]))
    if record.get("anti_pattern"):
        parts.append(f"  Avoid: {record['anti_pattern']}")
    if record.get("recovery_status") or record.get("source_failure_ids") or record.get("source_success_ids"):
        recovery_bits = []
        if record.get("recovery_status"):
            recovery_bits.append(f"status={record['recovery_status']}")
        if record.get("source_failure_ids"):
            recovery_bits.append("from_failures=" + ",".join(record["source_failure_ids"]))
        if record.get("source_success_ids"):
            recovery_bits.append("success_priors=" + ",".join(record["source_success_ids"]))
        parts.append("  Recovery: " + " | ".join(recovery_bits))
    if record.get("code_template"):
        parts.append(f"  Code: {record['code_template']}")
    return "\n".join(parts)
