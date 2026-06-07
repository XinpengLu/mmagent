from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional


ExperienceRecord = Dict[str, Any]
ExperienceLibrary = Dict[str, ExperienceRecord]

EXPERIENCE_LIBRARY_SCHEMA_VERSION = 1
VALID_KINDS = {"success", "failure"}
REQUIRED_RECORD_FIELDS = (
    "id",
    "kind",
    "skill_ids",
    "task_tags",
    "tool_tags",
    "confidence",
    "content",
    "stats",
)
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
OPERATION_METADATA_KEYS = (
    "kind",
    "outcome",
    "skill_ids",
    "task_tags",
    "tool_tags",
    "confidence",
    "stats",
    "tool_sequence",
    "tool_plan",
    "validation_checks",
    "validation_check",
    "fallback_plan",
    *OPTIONAL_TEXT_FIELDS,
    *OPTIONAL_LIST_FIELDS,
)


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


def _as_text_list(value: Any, max_items: int = 6) -> List[str]:
    """Normalize short free-text lists used by executable memory fields."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        items = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text") or item.get("check") or item.get("step") or item.get("purpose")
            else:
                text = item
            text = str(text or "").strip()
            if text and text not in items:
                items.append(text)
            if len(items) >= max_items:
                break
        return items
    text = str(value).strip()
    return [text] if text else []


def _normalize_tool_sequence(value: Any, max_items: int = 5) -> List[Dict[str, str]]:
    """Normalize an ordered tool plan into [{tool, purpose, expected_output}]."""
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    sequence: List[Dict[str, str]] = []
    for item in raw_items:
        if isinstance(item, dict):
            tool = str(item.get("tool") or item.get("name") or "").strip()
            purpose = str(item.get("purpose") or item.get("action") or item.get("step") or "").strip()
            expected = str(item.get("expected_output") or item.get("output") or item.get("validation") or "").strip()
        else:
            tool = ""
            purpose = str(item or "").strip()
            expected = ""
        if not tool and not purpose:
            continue
        normalized = {
            "tool": tool or "tool",
            "purpose": purpose or "execute the selected tool-backed check",
        }
        if expected:
            normalized["expected_output"] = expected
        if normalized not in sequence:
            sequence.append(normalized)
        if len(sequence) >= max_items:
            break
    return sequence


def _normalize_evidence_rollouts(value: Any, max_items: int = 8) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        kept = {}
        for key in ("doc_id", "rollout_id", "rollout_idx", "traj_path", "accuracy_score", "success", "final_answer"):
            if key in item and item[key] is not None:
                kept[key] = item[key]
        if kept:
            result.append(kept)
        if len(result) >= max_items:
            break
    return result


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
    """Return retrievable text from a structured record or generated operation."""
    if isinstance(value, dict):
        for key in ("content", "experience", "text"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return _clean_content_text(text)
        nested = value.get("experience")
        if isinstance(nested, dict):
            return get_experience_content(nested)
        return ""
    return ""


def get_experience_kind(value: Any, default: str = "success") -> str:
    if isinstance(value, dict):
        return _normalize_kind(value.get("kind") or value.get("outcome"), default=default)
    return default


def _is_canonical_experience_record(value: Any) -> bool:
    """Return True if value already has the canonical v1 runtime shape."""
    if not isinstance(value, dict):
        return False
    if any(field not in value for field in REQUIRED_RECORD_FIELDS):
        return False
    if value.get("kind") not in VALID_KINDS:
        return False
    if not isinstance(value.get("content"), str):
        return False
    if not isinstance(value.get("skill_ids"), list):
        return False
    if not isinstance(value.get("task_tags"), list):
        return False
    if not isinstance(value.get("tool_tags"), list):
        return False
    if not isinstance(value.get("confidence"), (int, float)):
        return False
    if not isinstance(value.get("stats"), dict):
        return False
    return True


def _experience_mapping_from_payload(raw: Any) -> Dict[str, Any]:
    """Extract an id->record mapping from canonical payloads or runtime mappings."""
    if not isinstance(raw, dict):
        raise TypeError("Experience library must be a dict.")

    if "schema_version" in raw or "experiences" in raw:
        if raw.get("schema_version") != EXPERIENCE_LIBRARY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported experience library schema_version={raw.get('schema_version')!r}; "
                f"expected {EXPERIENCE_LIBRARY_SCHEMA_VERSION}."
            )
        payload = raw.get("experiences")
        if isinstance(payload, dict):
            return payload
        raise ValueError("Canonical experience library must contain an 'experiences' object.")

    # Runtime mapping: {exp_id: ExperienceRecord}.
    return raw


LOW_INFORMATION_SUCCESS_PATTERNS = [
    r"\bsimilar visual reasoning problem\b",
    r"\bverified tool-backed solution\b",
    r"\breuse (?:the )?successful\b",
    r"\bsuccessful (?:image-analysis|code|tool) path\b",
    r"\btool-derived (?:visual or numeric )?evidence\b",
    r"\bdirectly supports the proposed answer\b",
]


def mark_low_information_success(record: ExperienceRecord) -> ExperienceRecord:
    """Down-weight generic success summaries that are not executable policies."""
    if not isinstance(record, dict) or get_experience_kind(record) != "success":
        return record
    content = get_experience_content(record).lower()
    if not any(re.search(pattern, content) for pattern in LOW_INFORMATION_SUCCESS_PATTERNS):
        return record

    updated = deepcopy(record)
    stats = dict(updated.get("stats") or {})
    stats["low_information_success"] = True
    updated["stats"] = stats
    updated["confidence"] = min(_as_float(updated.get("confidence"), 0.7), 0.35)
    return updated


def normalize_experience_record(
    exp_id: str,
    value: Any,
    *,
    default_kind: str = "success",
    source: Optional[Dict[str, Any]] = None,
) -> ExperienceRecord:
    """Coerce one generated or constructed record into the canonical v1 shape."""
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
    normalized_stats = {
        "support_count": int(stats.get("support_count", 1) or 1),
        "used_count": int(stats.get("used_count", 0) or 0),
        "success_after_used": int(stats.get("success_after_used", 0) or 0),
        "failure_after_used": int(stats.get("failure_after_used", 0) or 0),
    }
    # Keep validation/selection metadata added by the validation loop.
    for key, value in stats.items():
        if key in normalized_stats:
            continue
        if isinstance(value, (int, float, str, bool)):
            normalized_stats[key] = value

    record = {
        "id": record_id,
        "kind": kind,
        "skill_ids": _as_list(data.get("skill_ids")),
        "task_tags": _as_list(data.get("task_tags")),
        "tool_tags": _as_list(data.get("tool_tags")),
        "confidence": _as_float(data.get("confidence"), confidence_default),
        "content": content,
        "stats": normalized_stats,
    }
    tool_sequence = _normalize_tool_sequence(data.get("tool_sequence") or data.get("tool_plan"))
    if tool_sequence:
        record["tool_sequence"] = tool_sequence
    validation_checks = _as_text_list(data.get("validation_checks") or data.get("validation_check"))
    if validation_checks:
        record["validation_checks"] = validation_checks
    fallback_plan = _as_text_list(data.get("fallback_plan"))
    if fallback_plan:
        record["fallback_plan"] = fallback_plan
    evidence_rollouts = _normalize_evidence_rollouts(
        data.get("evidence_rollouts") or (source or {}).get("evidence_rollouts")
    )
    if evidence_rollouts:
        record["evidence_rollouts"] = evidence_rollouts

    for optional_key in OPTIONAL_TEXT_FIELDS:
        val = data.get(optional_key)
        if isinstance(val, str) and val.strip():
            record[optional_key] = val.strip()
    for optional_key in OPTIONAL_LIST_FIELDS:
        vals = _as_list(data.get(optional_key))
        if vals:
            record[optional_key] = vals
    return record


def normalize_experience_library(raw: Any) -> ExperienceLibrary:
    """Validate and normalize an in-memory runtime mapping."""
    raw = _experience_mapping_from_payload(raw)
    if not raw:
        return {}

    normalized: ExperienceLibrary = {}
    bad_ids = []
    for exp_id, value in raw.items():
        if not _is_canonical_experience_record(value):
            bad_ids.append(str(exp_id))
            continue
        normalized[str(exp_id)] = normalize_experience_record(str(exp_id), value)
    if bad_ids:
        preview = ", ".join(bad_ids[:8])
        suffix = "..." if len(bad_ids) > 8 else ""
        raise ValueError(f"Non-canonical experience records: {preview}{suffix}")
    return normalized


def _load_experience_library_payload(raw: Any) -> ExperienceLibrary:
    """Strictly parse the canonical on-disk experience library payload."""
    if not isinstance(raw, dict):
        raise TypeError("Experience library JSON must be an object.")
    if raw.get("schema_version") != EXPERIENCE_LIBRARY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported experience library schema_version={raw.get('schema_version')!r}; "
            f"expected {EXPERIENCE_LIBRARY_SCHEMA_VERSION}."
        )
    if not isinstance(raw.get("experiences"), dict):
        raise ValueError("Experience library JSON must contain an 'experiences' object.")
    return normalize_experience_library(raw["experiences"])


def _serialize_experience_library(experiences: Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical v1 JSON payload written to disk."""
    normalized = normalize_experience_library(experiences)
    records = {
        exp_id: normalize_experience_record(exp_id, record)
        for exp_id, record in normalized.items()
    }
    return {
        "schema_version": EXPERIENCE_LIBRARY_SCHEMA_VERSION,
        "experiences": records,
    }


def _compact_experience_record(value: Any, exp_id: str = "") -> ExperienceRecord:
    """Serialize a compact record for debug files and prompt-adjacent artifacts."""
    record = normalize_experience_record(exp_id, value)
    compact: ExperienceRecord = {
        "kind": record["kind"],
        "content": record["content"],
    }
    for key in ("task_tags", "tool_tags", "skill_ids"):
        if record.get(key):
            compact[key] = record[key]
    if record.get("confidence") is not None:
        compact["confidence"] = record["confidence"]
    if record.get("stats"):
        compact["stats"] = record["stats"]
    for key in (
        "tool_sequence",
        "validation_checks",
        "fallback_plan",
        *OPTIONAL_TEXT_FIELDS,
        *OPTIONAL_LIST_FIELDS,
    ):
        if record.get(key):
            compact[key] = record[key]
    if record.get("evidence_rollouts"):
        compact["evidence_rollouts"] = record["evidence_rollouts"]
    return compact


def _compact_experience_operation(operation: Dict[str, Any]) -> Dict[str, Any]:
    """Compact an add/modify operation for debug files."""
    if not isinstance(operation, dict):
        return operation
    compact = {"option": operation.get("option", "add")}
    if operation.get("modified_from"):
        compact["modified_from"] = str(operation["modified_from"])
    compact["experience"] = _compact_experience_record(operation.get("experience", operation))
    return compact


def _normalize_experience_operation(
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
        for key in OPERATION_METADATA_KEYS:
            if key not in data and key in operation:
                data[key] = operation[key]
        # Some callers add benchmark metadata at the operation level after an
        # LLM has produced an `experience` payload. Merge list-like metadata
        # instead of letting the nested payload silently drop it.
        for list_key in ("task_tags", "tool_tags", "skill_ids"):
            if list_key in operation:
                merged: List[str] = []
                for item in _as_list(data.get(list_key)) + _as_list(operation.get(list_key)):
                    if item not in merged:
                        merged.append(item)
                data[list_key] = merged
    else:
        data = {
            "kind": operation.get("kind") or operation.get("outcome") or default_kind,
            "skill_ids": operation.get("skill_ids", []),
            "task_tags": operation.get("task_tags", []),
            "tool_tags": operation.get("tool_tags", []),
            "confidence": operation.get("confidence"),
            "content": operation.get("content") or operation.get("experience") or operation.get("exp") or operation.get("text"),
            "stats": operation.get("stats", {}),
            "tool_sequence": operation.get("tool_sequence") or operation.get("tool_plan"),
            "validation_checks": operation.get("validation_checks") or operation.get("validation_check"),
            "fallback_plan": operation.get("fallback_plan"),
            "code_template": operation.get("code_template"),
            "anti_pattern": operation.get("anti_pattern"),
            "recovery_status": operation.get("recovery_status"),
            "source_failure_ids": operation.get("source_failure_ids"),
            "source_success_ids": operation.get("source_success_ids"),
            "recovery_source_ids": operation.get("recovery_source_ids"),
            "source_validation_stage": operation.get("source_validation_stage"),
            "source_validation_sample_id": operation.get("source_validation_sample_id"),
        }

    record = normalize_experience_record("", data, default_kind=default_kind, source=source)
    if not record["content"]:
        return None

    normalized = {"option": option, "experience": record}
    if option == "modify" and operation.get("modified_from"):
        normalized["modified_from"] = str(operation["modified_from"])
    return normalized


def _make_record_for_id(exp_id: str, operation: Any) -> ExperienceRecord:
    payload = operation.get("experience", operation) if isinstance(operation, dict) else operation
    return normalize_experience_record(exp_id, payload)


def _merge_sorted_record_list(records: List[ExperienceRecord], key: str) -> List[str]:
    return sorted({tag for record in records for tag in record.get(key, [])})


def _append_unique_until_limit(
    target: List[Any],
    values: Iterable[Any],
    limit: int,
    *,
    require_dict: bool = False,
) -> None:
    for value in values or []:
        if (not require_dict or isinstance(value, dict)) and value not in target:
            target.append(value)
        if len(target) >= limit:
            break


def _append_unique_texts(target: List[str], values: Iterable[Any]) -> None:
    for value in values or []:
        text = str(value).strip()
        if text and text not in target:
            target.append(text)


def _record_stats(record: ExperienceRecord) -> Dict[str, Any]:
    stats = record.get("stats")
    return stats if isinstance(stats, dict) else {}


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
        _append_unique_until_limit(merged["tool_sequence"], record.get("tool_sequence", []), 5, require_dict=True)
        _append_unique_until_limit(merged["validation_checks"], record.get("validation_checks", []), 6)
        _append_unique_until_limit(merged["fallback_plan"], record.get("fallback_plan", []), 4)
        _append_unique_until_limit(merged["evidence_rollouts"], record.get("evidence_rollouts", []), 8, require_dict=True)
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
        stats = _record_stats(record)
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
        status = str(record.get("recovery_status") or stats.get("recovery_status") or "").strip()
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
    merged_stats = {
        "support_count": sum(_record_stats(record).get("support_count", 1) for record in records),
        "used_count": sum(_record_stats(record).get("used_count", 0) for record in records),
        "success_after_used": sum(_record_stats(record).get("success_after_used", 0) for record in records),
        "failure_after_used": sum(_record_stats(record).get("failure_after_used", 0) for record in records),
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
        status = _merged_recovery_status(recovery["recovery_statuses"])
        if status:
            merged_stats["recovery_status"] = status
    return merged_stats


def _merge_record_metadata(exp_id: str, records: Iterable[Any], content: str) -> ExperienceRecord:
    """Create one structured record from several same-kind records and merged content."""
    normalized = [normalize_experience_record("", record) for record in records if get_experience_content(record)]
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
        "confidence": sum(record.get("confidence", 0.5) for record in normalized) / len(normalized),
        "content": content,
        "stats": _merge_record_stats(normalized, recovery),
        **sequence_fields,
    }
    if recovery["recovered_from_failure"] and payload["stats"].get("recovery_status"):
        payload["recovery_status"] = payload["stats"]["recovery_status"]
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
