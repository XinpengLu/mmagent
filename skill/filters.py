"""Quality gates and metadata filters for memory injection."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

from experience.schema import (
    ExperienceLibrary,
    normalize_experience_library,
)
from utils.env import env_flag, env_float, env_int
from .utils import (
    metadata_field_conflicts,
    metadata_field_matches,
    metadata_fields,
    record_metadata_fields,
)


def _stats_of(item: Dict[str, Any]) -> Dict[str, Any]:
    stats = item.get("stats") if isinstance(item, dict) else {}
    return stats if isinstance(stats, dict) else {}


def _support_counts_of(item: Dict[str, Any]) -> Dict[str, Any]:
    counts = item.get("support_counts") if isinstance(item, dict) else {}
    return counts if isinstance(counts, dict) else {}


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def _validation_scores(stats: Dict[str, Any], average_default: Any = 0.0) -> Tuple[float, float]:
    try:
        return (
            float(stats.get("validation_average_score", average_default) or 0.0),
            float(stats.get("validation_pass_rate", 0.0) or 0.0),
        )
    except (TypeError, ValueError):
        return 0.0, 0.0


def _env_float_or_default(name: str, default: float) -> float:
    try:
        return env_float(name, default)
    except (TypeError, ValueError):
        return default


def allow_unvalidated_recovery_inference() -> bool:
    """Whether unverified failure-to-success candidates can be injected."""
    if env_flag("ENABLE_UNVALIDATED_RECOVERY_INFERENCE"):
        return True
    return env_flag("MMAGENT_VALIDATION_PHASE") or env_flag("MMAGENT_TRAIN_PHASE")


def experience_unvalidated_recovery(record: Dict[str, Any]) -> bool:
    """True for recovery candidates that have not passed validation yet."""
    if not isinstance(record, dict):
        return False
    stats = _stats_of(record)
    status = str(record.get("recovery_status") or stats.get("recovery_status") or "").strip()
    return status == "candidate_unverified"


def skill_unvalidated_recovery(item: Dict[str, Any]) -> bool:
    """True for recovery skills that should wait for explicit validation."""
    if not isinstance(item, dict):
        return False
    status = str(item.get("recovery_status") or "").strip()
    return item.get("skill_type") == "recovery_action_skill" and status == "candidate_unverified"


def recovery_has_train_support(item: Dict[str, Any]) -> bool:
    """Return True when an unverified recovery has positive supervised-train use."""
    if not isinstance(item, dict):
        return False
    stats = _stats_of(item)
    if not (
        skill_unvalidated_recovery(item)
        or experience_unvalidated_recovery(item)
        or str(item.get("recovery_status") or stats.get("recovery_status") or "") == "candidate_unverified"
    ):
        return False
    if not env_flag("ENABLE_TRAIN_SUPPORTED_RECOVERY_VALIDATION_PROBE", True):
        return False
    if not env_flag("MMAGENT_VALIDATION_PHASE"):
        return False
    used = int(stats.get("used_count", 0) or 0)
    success = int(stats.get("success_after_used", 0) or 0)
    failure = int(stats.get("failure_after_used", 0) or 0)
    judged = success + failure
    min_uses = env_int("TRAIN_SUPPORTED_RECOVERY_MIN_USES", 2)
    min_rate = env_float("TRAIN_SUPPORTED_RECOVERY_MIN_SUCCESS_RATE", 0.50)
    return used >= min_uses and judged >= min_uses and (success / max(1, judged)) >= min_rate


def _explicit_task_count(item: Dict[str, Any]) -> int:
    fields = metadata_fields(item.get("task_tags", []) or [])
    return len(fields.get("task", set()) or set())


def _singleton_action_skill_allowed(item: Dict[str, Any]) -> bool:
    """Allow one-off action skills only when they are narrow and confident."""
    if item.get("skill_type") != "action_skill":
        return True
    support_counts = _support_counts_of(item)
    success_support = int(support_counts.get("success", 0) or 0)
    min_action_support = env_int("ACTION_SKILL_MIN_SUCCESS_SUPPORT", 2)
    if success_support >= min_action_support:
        return True
    if _explicit_task_count(item) != 1:
        return False
    confidence = _float_value(item.get("confidence"), 0.0)
    min_conf = env_float("SINGLETON_ACTION_SKILL_MIN_CONFIDENCE", 0.65)
    return confidence >= min_conf


def experience_failed_validation(record: Dict[str, Any]) -> bool:
    """True when validation says this memory should not support inference/skills."""
    if record.get("recovery_status") == "rejected":
        return True

    stats = _stats_of(record)
    if stats.get("recovery_status") == "rejected":
        return True

    val_used = int(stats.get("validation_used_count", 0) or 0)
    recovery_status = record.get("recovery_status") or stats.get("recovery_status")
    is_recovery = bool(stats.get("recovered_from_failure")) or bool(recovery_status)
    if is_recovery:
        min_validation_uses = env_int("MIN_VALIDATION_USES_TO_SKIP_RECOVERY_EXPERIENCE", 3)
    elif record.get("kind") == "failure":
        min_validation_uses = env_int("MIN_VALIDATION_USES_TO_SKIP_FAILURE_EXPERIENCE", 1)
    else:
        min_validation_uses = env_int("MIN_VALIDATION_USES_TO_SKIP_EXPERIENCE", 2)
    if val_used < min_validation_uses:
        return False

    val_avg, val_pass = _validation_scores(stats)
    max_bad_validation_avg = env_float("BAD_EXPERIENCE_MAX_VALIDATION_AVG", 0.01)
    max_bad_validation_pass = env_float("BAD_EXPERIENCE_MAX_VALIDATION_PASS", 0.0)
    return val_avg <= max_bad_validation_avg and val_pass <= max_bad_validation_pass


def _experience_unsupported_success(record: Dict[str, Any]) -> bool:
    """True for success-shaped memories whose attached rollout evidence all failed."""
    if not isinstance(record, dict) or record.get("kind") != "success":
        return False

    stats = _stats_of(record)
    if float(stats.get("validation_pass_rate", 0.0) or 0.0) > 0.0:
        return False

    evidence = [r for r in record.get("evidence_rollouts", []) or [] if isinstance(r, dict)]
    judged = [
        r for r in evidence
        if "success" in r or "accuracy_score" in r
    ]
    if not judged:
        return False
    return not any(bool(r.get("success")) or float(r.get("accuracy_score", 0.0) or 0.0) > 0.0 for r in judged)


def experience_low_validation_quality(record: Dict[str, Any]) -> bool:
    """True when a memory has been tried on validation but remains too weak.

    `experience_failed_validation` catches hard failures. This softer gate
    handles memories that are not completely wrong but repeatedly add noisy
    tool plans. Keeping this separate makes it easy to use weaker memories as
    background text while preventing them from driving new skills/actions.
    """
    if experience_failed_validation(record):
        return True
    if _experience_unsupported_success(record):
        return True

    stats = _stats_of(record)
    val_used = int(stats.get("validation_used_count", 0) or 0)
    if val_used <= 0:
        return False

    recovery_status = record.get("recovery_status") or stats.get("recovery_status")
    is_recovery = bool(stats.get("recovered_from_failure")) or bool(recovery_status)
    if is_recovery:
        min_uses = env_int("MIN_VALIDATION_USES_TO_DEMOTE_RECOVERY_EXPERIENCE", 2)
    else:
        min_uses = env_int("MIN_VALIDATION_USES_TO_DEMOTE_EXPERIENCE", 2)
    if val_used < min_uses:
        return False

    val_avg, val_pass = _validation_scores(stats)
    min_avg = env_float("LOW_QUALITY_EXPERIENCE_MIN_VALIDATION_AVG", 0.12)
    max_pass = env_float("LOW_QUALITY_EXPERIENCE_MAX_PASS_RATE", 0.25)
    return val_avg < min_avg and val_pass <= max_pass


def skill_low_validation_quality(item: Dict[str, Any]) -> bool:
    """True when validation has shown that a skill is consistently weak."""
    if not isinstance(item, dict):
        return False

    stats = _stats_of(item)
    val_used = int(stats.get("validation_used_count", 0) or 0)
    if val_used <= 0:
        return False

    if item.get("skill_type") == "recovery_action_skill":
        min_uses = env_int("MIN_VALIDATION_USES_TO_DEMOTE_RECOVERY_SKILL", 2)
    else:
        min_uses = env_int("MIN_VALIDATION_USES_TO_DEMOTE_SKILL", 3)
    if val_used < min_uses:
        return False

    val_avg, val_pass = _validation_scores(stats, item.get("validation_quality", 0.0))
    min_avg = env_float("LOW_QUALITY_SKILL_MIN_VALIDATION_AVG", 0.12)
    max_pass = env_float("LOW_QUALITY_SKILL_MAX_PASS_RATE", 0.25)
    return val_avg < min_avg and val_pass <= max_pass


def _skill_hard_blocked(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("disabled") or item.get("recovery_status") == "rejected":
        return True
    if skill_low_validation_quality(item):
        return True
    return (
        skill_unvalidated_recovery(item)
        and not allow_unvalidated_recovery_inference()
        and not recovery_has_train_support(item)
    )


def _usage_adjusted_skill_weight(item: Dict[str, Any], weight: float) -> float:
    stats = _stats_of(item)
    used_count = int(stats.get("used_count", 0) or 0)
    success_after_used = int(stats.get("success_after_used", 0) or 0)
    failure_after_used = int(stats.get("failure_after_used", 0) or 0)
    judged_uses = success_after_used + failure_after_used
    if not (used_count and judged_uses):
        return weight

    usage_rate = success_after_used / max(1, judged_uses)
    min_usage_uses = env_int("MIN_MEMORY_USES_TO_DOWNWEIGHT_SKILL", 4)
    skip_rate = env_float("MEMORY_USAGE_SKIP_SUCCESS_RATE", 0.25)
    weak_rate = env_float("MEMORY_USAGE_WEAK_SUCCESS_RATE", 0.45)
    strong_rate = env_float("MEMORY_USAGE_STRONG_SUCCESS_RATE", 0.65)
    if used_count >= min_usage_uses and usage_rate <= skip_rate:
        return 0.0
    if used_count >= min_usage_uses and usage_rate < weak_rate:
        return weight * 0.35
    if used_count >= min_usage_uses and usage_rate >= strong_rate:
        return weight * 1.15
    return weight


def _validation_adjusted_skill_weight(item: Dict[str, Any], weight: float) -> float:
    stats = _stats_of(item)
    val_used = int(stats.get("validation_used_count", 0) or 0)
    if not val_used:
        return weight

    quality = stats.get("validation_average_score", item.get("validation_quality", 0.5))
    quality = max(0.0, min(1.0, _float_value(quality, 0.5)))
    min_skip_uses = env_int("MIN_VALIDATION_USES_TO_SKIP_SKILL", 3)
    if val_used >= min_skip_uses and quality <= 0.05:
        return 0.0
    weight *= 0.65 + quality
    if quality <= 0.05:
        weight *= 0.25 if val_used >= 1 else 0.55
    return weight


def _failure_guardrail_weight(item: Dict[str, Any], weight: float) -> float:
    stats = _stats_of(item)
    val_used = int(stats.get("validation_used_count", 0) or 0)
    support_counts = _support_counts_of(item)
    success_support = int(support_counts.get("success", 0) or 0)
    failure_support = int(support_counts.get("failure", 0) or 0)
    if val_used == 0 and success_support == 0 and failure_support <= 1:
        return weight * 0.35
    return weight * 0.75


def _action_skill_weight(item: Dict[str, Any], weight: float) -> float:
    stats = _stats_of(item)
    support_counts = _support_counts_of(item)
    success_support = int(support_counts.get("success", 0) or 0)
    min_action_support = env_int("ACTION_SKILL_MIN_SUCCESS_SUPPORT", 2)
    if (
        item.get("skill_type") == "action_skill"
        and success_support < min_action_support
        and int(stats.get("validation_used_count", 0) or 0) == 0
        and not _singleton_action_skill_allowed(item)
    ):
        return 0.0
    if item.get("skill_type") == "action_skill" and success_support < min_action_support:
        weight *= env_float("SINGLETON_ACTION_SKILL_SELECTION_WEIGHT", 0.55)

    confidence = _float_value(item.get("confidence"), 0.7)
    if confidence < 0.45:
        weight *= 0.45
    elif confidence < 0.60:
        weight *= 0.75
    return weight * 1.15


def _recovery_skill_weight(item: Dict[str, Any], weight: float) -> float:
    if item.get("skill_type") != "recovery_action_skill":
        return weight

    stats = _stats_of(item)
    val_used = int(stats.get("validation_used_count", 0) or 0)
    status = str(item.get("recovery_status") or "candidate_unverified")
    if status == "candidate_unverified":
        if item.get("pending_validation_revision"):
            return weight * env_float("RECOVERY_PENDING_REVISION_SELECTION_WEIGHT", 0.9)
        if val_used > 0:
            return weight * env_float("RECOVERY_CANDIDATE_WITH_FEEDBACK_SELECTION_WEIGHT", 0.35)
        return weight * env_float("RECOVERY_CANDIDATE_SELECTION_WEIGHT", 0.25)
    if status == "validated":
        return weight * env_float("RECOVERY_VALIDATED_SELECTION_WEIGHT", 1.15)
    return weight


def _type_adjusted_skill_weight(item: Dict[str, Any], weight: float) -> float:
    if item.get("skill_type") == "failure_guardrail":
        return _failure_guardrail_weight(item, weight)
    weight = _action_skill_weight(item, weight)
    return _recovery_skill_weight(item, weight)


def skill_selection_weight(item: Dict[str, Any]) -> float:
    """Convert validation feedback into a soft retrieval weight."""
    if not isinstance(item, dict):
        return 1.0
    if _skill_hard_blocked(item):
        return 0.0

    weight = float(item.get("selection_weight", 1.0) or 1.0)
    weight = _usage_adjusted_skill_weight(item, weight)
    if weight <= 0.0:
        return 0.0
    weight = _validation_adjusted_skill_weight(item, weight)
    if weight <= 0.0:
        return 0.0
    weight = _type_adjusted_skill_weight(item, weight)
    if item.get("needs_refine"):
        weight *= 0.9
    return max(0.0, weight)


def _skill_usage_rate(item: Dict[str, Any]) -> Tuple[int, Optional[float]]:
    """Return historical use count and observed success rate, when available."""
    stats = _stats_of(item)
    used_count = int(stats.get("used_count", 0) or 0)
    success_after_used = int(stats.get("success_after_used", 0) or 0)
    failure_after_used = int(stats.get("failure_after_used", 0) or 0)
    judged_uses = success_after_used + failure_after_used
    if not used_count or not judged_uses:
        return used_count, None
    return used_count, success_after_used / max(1, judged_uses)


def _skill_has_poor_usage(item: Dict[str, Any]) -> bool:
    used_count, usage_rate = _skill_usage_rate(item)
    if usage_rate is None:
        return False
    min_usage_uses = env_int("MIN_MEMORY_USES_TO_DOWNWEIGHT_SKILL", 4)
    skip_rate = env_float("MEMORY_USAGE_SKIP_SUCCESS_RATE", 0.25)
    return used_count >= min_usage_uses and usage_rate <= skip_rate


def _basic_skill_gate(metadata: Dict[str, Any]) -> Tuple[bool, str]:
    if metadata.get("recovery_status") == "rejected":
        return False, "rejected_recovery_skill"
    if skill_unvalidated_recovery(metadata) and not allow_unvalidated_recovery_inference():
        return False, "unvalidated_recovery_skill"
    if _skill_has_poor_usage(metadata):
        return False, "poor_usage_history"
    if skill_low_validation_quality(metadata):
        return False, "low_validation_quality"
    return True, ""


def _recovery_retry_gate(metadata: Dict[str, Any]) -> Tuple[bool, str]:
    stats = _stats_of(metadata)
    val_used = int(stats.get("validation_used_count", 0) or 0)
    if not (
        skill_unvalidated_recovery(metadata)
        and val_used > 0
        and not metadata.get("pending_validation_revision")
    ):
        return True, ""
    val_avg = _float_value(stats.get("validation_average_score", metadata.get("validation_quality", 0.0)), 0.0)
    min_retry_uses = env_int("RECOVERY_CANDIDATE_RETRY_MIN_VALIDATION_USES", 1)
    min_retry_avg = env_float("RECOVERY_CANDIDATE_RETRY_MIN_AVERAGE", 0.20)
    if val_used >= min_retry_uses and val_avg < min_retry_avg:
        return False, "unverified_recovery_low_validation_without_revision"
    return True, ""


def _singleton_action_gate(metadata: Dict[str, Any]) -> Tuple[bool, str]:
    stats = _stats_of(metadata)
    val_used = int(stats.get("validation_used_count", 0) or 0)
    support_counts = _support_counts_of(metadata)
    success_support = int(support_counts.get("success", 0) or 0)
    min_action_support = env_int("ACTION_SKILL_MIN_SUCCESS_SUPPORT", 2)
    if (
        metadata.get("skill_type") == "action_skill"
        and success_support < min_action_support
        and val_used == 0
        and not _singleton_action_skill_allowed(metadata)
    ):
        return False, "singleton_unvalidated_action_skill"
    return True, ""


def _failed_validation_gate(metadata: Dict[str, Any]) -> Tuple[bool, str]:
    stats = _stats_of(metadata)
    val_used = int(stats.get("validation_used_count", 0) or 0)
    if not val_used:
        return True, ""
    min_val_uses = env_int("MIN_VALIDATION_USES_TO_SKIP_SKILL", 3)
    disable_avg = env_float("VALIDATION_SKIP_AVERAGE_THRESHOLD", 0.01)
    disable_pass = env_float("VALIDATION_SKIP_PASS_THRESHOLD", 0.0)
    val_avg, val_pass = _validation_scores(stats, metadata.get("validation_quality", 0.0))
    if val_used >= min_val_uses and val_avg <= disable_avg and val_pass <= disable_pass:
        return False, "failed_validation_history"
    return True, ""


def _metadata_conflict_gate(
    skill: Dict[str, Any],
    metadata: Dict[str, Any],
    query_metadata_tags: Optional[Iterable[str]],
) -> Tuple[bool, str]:
    query_fields = metadata_fields(query_metadata_tags or [])
    candidate_fields = record_metadata_fields(metadata)
    if not (query_fields and candidate_fields):
        return True, ""

    if metadata_field_conflicts(query_fields, candidate_fields, "data_source"):
        return False, "data_source_mismatch"
    allow_raw = env_float("MEMORY_SELECTION_MISMATCH_ALLOW_RAW_SCORE", 0.38)
    raw_score = float(skill.get("raw_score", 0.0) or 0.0)
    for field in ("task", "category", "prompt_category", "subtask"):
        query_values = query_fields.get(field)
        candidate_values = candidate_fields.get(field)
        if field == "task" and query_values and candidate_values and len(candidate_values) > 1:
            return False, "mixed_task_skill"
        conflicts = metadata_field_conflicts(query_fields, candidate_fields, field)
        if field == "task" and conflicts:
            return False, "task_mismatch"
        if conflicts and raw_score < allow_raw:
            return False, f"{field}_mismatch"
    return True, ""


def _weak_one_off_skill(skill: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
    score = float(skill.get("score", 0.0) or 0.0)
    raw_score = float(skill.get("raw_score", 0.0) or 0.0)
    min_score = env_float("MEMORY_SELECTION_MIN_SKILL_SCORE", 0.06)
    min_raw_score = env_float("MEMORY_SELECTION_MIN_RAW_SKILL_SCORE", 0.18)
    support_counts = _support_counts_of(metadata)
    support_total = int(support_counts.get("success", 0) or 0) + int(support_counts.get("failure", 0) or 0)
    return support_total <= 1 and score < min_score and raw_score < min_raw_score


def selection_gate_allows_skill(
    skill: Dict[str, Any],
    query_metadata_tags: Optional[Iterable[str]],
) -> Tuple[bool, str]:
    """Conservative post-rerank gate to avoid broad low-quality skills."""
    metadata = skill.get("metadata", {}) if isinstance(skill, dict) else {}
    for gate in (
        lambda: _basic_skill_gate(metadata),
        lambda: _recovery_retry_gate(metadata),
        lambda: _singleton_action_gate(metadata),
        lambda: _failed_validation_gate(metadata),
        lambda: _metadata_conflict_gate(skill, metadata, query_metadata_tags),
    ):
        allowed, reason = gate()
        if not allowed:
            return False, reason

    if _weak_one_off_skill(skill, metadata):
        return False, "weak_one_off_skill"
    return True, ""


def _experience_metadata_matches_query(record: Dict[str, Any], query_metadata_tags: Optional[Iterable[str]]) -> bool:
    """Require explicit metadata conflicts to be absent before keeping direct failure memories."""
    query_fields = metadata_fields(query_metadata_tags or [])
    if not query_fields:
        return True
    record_fields = record_metadata_fields(record)
    if not record_fields:
        return False
    if metadata_field_conflicts(query_fields, record_fields, "data_source"):
        return False
    return True


def _experience_task_metadata_conflicts_query(record: Dict[str, Any], query_metadata_tags: Optional[Iterable[str]]) -> bool:
    """Drop memories that explicitly belong to a different task slice."""
    query_fields = metadata_fields(query_metadata_tags or [])
    record_fields = record_metadata_fields(record)
    return metadata_field_conflicts(query_fields, record_fields, "task")


def _experience_task_metadata_ambiguous(record: Dict[str, Any], query_metadata_tags: Optional[Iterable[str]]) -> bool:
    """Drop merged memories that span multiple explicit benchmark task slices."""
    query_fields = metadata_fields(query_metadata_tags or [])
    record_fields = record_metadata_fields(record)
    query_tasks = query_fields.get("task")
    record_tasks = record_fields.get("task")
    return bool(query_tasks and record_tasks and len(record_tasks) > 1)


def _experience_metadata_strongly_matches_query(record: Dict[str, Any], query_metadata_tags: Optional[Iterable[str]]) -> bool:
    """Check whether a search guardrail is from the same benchmark/domain slice."""
    query_fields = metadata_fields(query_metadata_tags or [])
    record_fields = record_metadata_fields(record)
    if not query_fields or not record_fields:
        return False
    if metadata_field_conflicts(query_fields, record_fields, "data_source"):
        return False
    for field in ("category", "domain", "subdomain", "subtask", "prompt_category"):
        if metadata_field_matches(query_fields, record_fields, field):
            return True
    return metadata_field_matches(query_fields, record_fields, "data_source")


def _is_actionable_failure_memory(
    record: Dict[str, Any],
    score: Optional[float],
    query_metadata_tags: Optional[Iterable[str]],
) -> bool:
    """Keep only compact failure guardrails that can guide local visual/computational checks."""
    if record.get("kind") != "failure":
        return False
    if not _experience_metadata_matches_query(record, query_metadata_tags):
        return False

    min_score = _env_float_or_default("FAILURE_ONLY_KEEP_MIN_SIMILARITY", 0.25)
    if score is not None and score < min_score:
        return False

    tool_tags = {str(tag).strip() for tag in record.get("tool_tags", []) or [] if str(tag).strip()}
    search_tools = {"web_search", "visit", "image_search"}
    local_tools = {"code_interpreter", "zoom"}
    has_local_tool = bool(tool_tags & local_tools)
    search_only = bool(tool_tags) and tool_tags <= search_tools

    has_guardrail_fields = bool(
        record.get("validation_checks")
        or record.get("fallback_plan")
        or record.get("anti_pattern")
        or record.get("tool_sequence")
    )
    if search_only:
        search_min_score = _env_float_or_default("SEARCH_FAILURE_ONLY_KEEP_MIN_SIMILARITY", 0.32)
        return (
            has_guardrail_fields
            and score is not None
            and score >= search_min_score
            and _experience_metadata_strongly_matches_query(record, query_metadata_tags)
        )
    if has_local_tool:
        return has_guardrail_fields
    if not tool_tags and has_guardrail_fields:
        return True
    return False


def filter_experiences_for_inference(
    experiences: ExperienceLibrary,
    allow_failure_only: bool = False,
    similarity_scores: Optional[Dict[str, float]] = None,
    query_metadata_tags: Optional[Iterable[str]] = None,
) -> ExperienceLibrary:
    """Keep memories that are likely to help inference rather than inject weak failure-only noise."""
    normalized = normalize_experience_library(experiences)
    if not normalized:
        return {}

    has_success = any(record.get("kind") == "success" for record in normalized.values())
    if not has_success and not allow_failure_only:
        similarity_scores = similarity_scores or {}
        candidates = []
        for exp_id, record in normalized.items():
            score = similarity_scores.get(exp_id)
            if _is_actionable_failure_memory(record, score, query_metadata_tags):
                candidates.append((exp_id, record, float(score) if score is not None else 0.0))
        if not candidates:
            return {}
        max_items = env_int("MAX_FAILURE_ONLY_EXPERIENCE_ITEMS", 2)
        candidates.sort(key=lambda item: item[2], reverse=True)
        normalized = {exp_id: record for exp_id, record, _ in candidates[:max(1, max_items)]}

    min_uses = env_int("MIN_FAILURE_EXPERIENCE_USES_TO_FILTER", 3)
    min_rate = env_float("FAILURE_EXPERIENCE_MIN_SUCCESS_RATE", 0.30)
    filtered: ExperienceLibrary = {}
    for exp_id, record in normalized.items():
        if _experience_task_metadata_conflicts_query(record, query_metadata_tags):
            continue
        if _experience_task_metadata_ambiguous(record, query_metadata_tags):
            continue
        if experience_unvalidated_recovery(record) and not allow_unvalidated_recovery_inference():
            continue
        if experience_failed_validation(record):
            continue
        if experience_low_validation_quality(record):
            continue
        stats = _stats_of(record)
        if record.get("kind") != "failure":
            filtered[exp_id] = record
            continue
        success_after_used = int(stats.get("success_after_used", 0) or 0)
        failure_after_used = int(stats.get("failure_after_used", 0) or 0)
        judged_uses = success_after_used + failure_after_used
        if judged_uses >= min_uses:
            success_rate = success_after_used / max(1, judged_uses)
            if success_rate < min_rate:
                continue
        filtered[exp_id] = record
    return filtered
