"""LLM-assisted failure-to-success recovery synthesis.

The code here intentionally avoids task-specific rules. It only filters obvious
system failures, samples a few failure/success candidates, asks an LLM whether
the failure is recoverable, and stores the proposed repair as an unverified
candidate experience. Validation feedback later promotes or disables it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from prompts.recovery_templates import RECOVERY_SYNTHESIS_PROMPT

from configs.runtime import runtime_setting
from utils.json_utils import parse_llm_json
from experience.schema import (
    ExperienceLibrary,
    format_experience_for_prompt,
    get_experience_content,
    normalize_experience_library,
    normalize_experience_record,
)
from .utils import metadata_values, task_metadata_tags


_SYSTEM_FAILURE_TERMS = (
    "api error",
    "upstream_error",
    "connection reset",
    "timeout",
    "too many open files",
    "kernel died",
    "empty response",
    "service temporarily unavailable",
    "unauthorized",
    "token expired",
)
def _stats_of(record: Dict[str, Any]) -> Dict[str, Any]:
    stats = record.get("stats") if isinstance(record, dict) else {}
    return stats if isinstance(stats, dict) else {}


def _record_text(record: Dict[str, Any]) -> str:
    parts = [
        get_experience_content(record),
        str(record.get("anti_pattern") or ""),
        " ".join(record.get("validation_checks") or []),
        " ".join(record.get("fallback_plan") or []),
    ]
    for step in record.get("tool_sequence") or []:
        if isinstance(step, dict):
            parts.append(str(step.get("purpose") or ""))
            parts.append(str(step.get("expected_output") or ""))
    return " ".join(parts).lower()


def _is_system_failure(record: Dict[str, Any]) -> bool:
    text = _record_text(record)
    return any(term in text for term in _SYSTEM_FAILURE_TERMS)


def _has_recovery_evidence(record: Dict[str, Any]) -> bool:
    return bool(
        record.get("tool_sequence")
        or record.get("code_template")
        or record.get("validation_checks")
        or record.get("fallback_plan")
        or record.get("anti_pattern")
    )


def _is_recoverable_candidate(record: Dict[str, Any]) -> bool:
    return (
        record.get("kind") == "failure"
        and not _is_system_failure(record)
        and _has_recovery_evidence(record)
    )


def _is_success_prior(record: Dict[str, Any]) -> bool:
    if record.get("kind") != "success":
        return False
    stats = _stats_of(record)
    if stats.get("recovered_from_failure") or stats.get("low_information_success"):
        return False
    return _has_recovery_evidence(record)


def _next_experience_id(experiences: ExperienceLibrary) -> str:
    max_id = 0
    for exp_id in experiences:
        text = str(exp_id)
        if text.startswith("E") and text[1:].isdigit():
            max_id = max(max_id, int(text[1:]))
    return f"E{max_id + 1}"


def _existing_recovery_pairs(experiences: ExperienceLibrary) -> set:
    pairs = set()
    for record in experiences.values():
        if not isinstance(record, dict):
            continue
        stats = _stats_of(record)
        if not stats.get("recovered_from_failure"):
            continue
        for failure_id in record.get("source_failure_ids") or []:
            for success_id in record.get("source_success_ids") or []:
                pairs.add((str(failure_id), str(success_id)))
    return pairs


def _recovery_counts_by_failure(experiences: ExperienceLibrary) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in experiences.values():
        if not isinstance(record, dict):
            continue
        stats = _stats_of(record)
        if not stats.get("recovered_from_failure"):
            continue
        for failure_id in record.get("source_failure_ids") or []:
            failure_id = str(failure_id)
            counts[failure_id] = counts.get(failure_id, 0) + 1
    return counts


def _safe_json_loads(text: str) -> Dict[str, Any]:
    data = parse_llm_json(text)
    if isinstance(data, list):
        return {"recoveries": data}
    return data if isinstance(data, dict) else {}


def _tag_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    a_tags = {str(tag).strip().lower() for tag in (a.get("task_tags") or []) + (a.get("tool_tags") or [])}
    b_tags = {str(tag).strip().lower() for tag in (b.get("task_tags") or []) + (b.get("tool_tags") or [])}
    return len(a_tags & b_tags)


def _metadata_compatible(failure: Dict[str, Any], success: Dict[str, Any]) -> bool:
    """Keep LLM recovery candidates within explicit benchmark task slices."""
    failure_tasks = metadata_values(failure, "task_")
    success_tasks = metadata_values(success, "task_")
    if failure_tasks and success_tasks and not (failure_tasks & success_tasks):
        return False
    failure_sources = metadata_values(failure, "data_source_")
    success_sources = metadata_values(success, "data_source_")
    if failure_sources and success_sources and not (failure_sources & success_sources):
        return False
    return True


def _select_success_candidates(
    failure: Dict[str, Any],
    successes: List[Tuple[str, Dict[str, Any]]],
    max_candidates: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Lightweight sampling only; the LLM makes the compatibility decision."""
    compatible_successes = [
        item for item in successes
        if _metadata_compatible(failure, item[1])
    ]
    if compatible_successes:
        successes = compatible_successes
    ranked = sorted(successes, key=lambda item: _tag_overlap(failure, item[1]), reverse=True)
    return ranked[:max(1, max_candidates)]


def _format_record(exp_id: str, record: Dict[str, Any]) -> str:
    return f"[{exp_id}]\n{format_experience_for_prompt(exp_id, record)}"


def _as_list(value: Any, limit: int = 8) -> List[str]:
    if value is None:
        return []
    raw = value if isinstance(value, list) else [value]
    out: List[str] = []
    for item in raw:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _compact_recovery_task_tags(
    proposal: Dict[str, Any],
    failure: Dict[str, Any],
    limit: int = 8,
) -> List[str]:
    """Use the failure task as the retrieval anchor; success prior supplies policy."""
    tags: List[str] = []
    for tag in task_metadata_tags(failure) + _as_list(proposal.get("task_tags"), limit):
        text = str(tag).strip().lower()
        if text and text not in tags:
            tags.append(text)
        if len(tags) >= limit:
            break
    if "recovery_from_failure" not in tags:
        tags.append("recovery_from_failure")
    return tags


def _normalize_tool_sequence(value: Any, limit: int = 5) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    out: List[Dict[str, str]] = []
    for step in value:
        if not isinstance(step, dict):
            continue
        tool = str(step.get("tool") or "").strip()
        purpose = str(step.get("purpose") or step.get("action") or "").strip()
        expected = str(step.get("expected_output") or step.get("validation") or "").strip()
        if not tool and not purpose:
            continue
        normalized = {"tool": tool or "tool", "purpose": purpose or "run the repaired tool step"}
        if expected:
            normalized["expected_output"] = expected
        out.append(normalized)
        if len(out) >= limit:
            break
    return out


class _RecoverySynthesizer:
    """Thin LLM service for turning failures into unverified recovery candidates."""

    def __init__(self, llm: Any, *, max_success_candidates: int = 4):
        self.llm = llm
        self.max_success_candidates = max(1, int(max_success_candidates or 4))
        self.max_tokens = 1024

    def propose_recovery(
        self,
        failure_experience: Dict[str, Any],
        success_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if self.llm is None or not success_candidates:
            return []

        failure_id = str(failure_experience.get("id") or "failure")
        prompt = RECOVERY_SYNTHESIS_PROMPT.format(
            failure_experience=_format_record(failure_id, failure_experience),
            success_candidates="\n\n".join(
                _format_record(str(candidate.get("id") or f"S{i}"), candidate)
                for i, candidate in enumerate(success_candidates[: self.max_success_candidates])
            ),
        )
        try:
            response = self.llm.chat(prompt, max_tokens=self.max_tokens, temperature=0.2).strip()
            parsed = _safe_json_loads(response)
        except Exception as exc:
            print(f"  [Failure Recovery] LLM recovery synthesis failed for {failure_id}: {exc}")
            return []

        proposals = parsed.get("recoveries") if isinstance(parsed.get("recoveries"), list) else [parsed]
        return [proposal for proposal in proposals if isinstance(proposal, dict) and proposal.get("recoverable") is True]


def _proposal_to_record(
    recovery_id: str,
    failure_id: str,
    failure: Dict[str, Any],
    proposal: Dict[str, Any],
    success_by_id: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    success_id = str(proposal.get("source_success_id") or "").strip()
    if not success_id or success_id not in success_by_id:
        return None

    content = str(proposal.get("content") or "").strip()
    if not content:
        return None

    success = success_by_id[success_id]
    if not _metadata_compatible(failure, success):
        return None
    task_tags = _compact_recovery_task_tags(proposal, failure)
    tool_tags = sorted(set(_as_list(proposal.get("tool_tags"), 6) + (failure.get("tool_tags") or []) + (success.get("tool_tags") or [])))

    return normalize_experience_record(
        recovery_id,
        {
            "kind": "success",
            "task_tags": task_tags,
            "tool_tags": tool_tags,
            "content": content,
            "tool_sequence": _normalize_tool_sequence(proposal.get("tool_sequence")) or success.get("tool_sequence") or [],
            "validation_checks": _as_list(proposal.get("validation_checks"), 6),
            "fallback_plan": _as_list(proposal.get("fallback_plan"), 5),
            "code_template": str(proposal.get("code_template") or success.get("code_template") or "").strip(),
            "anti_pattern": str(proposal.get("anti_pattern") or proposal.get("failure_reason") or failure.get("anti_pattern") or "").strip(),
            "recovery_status": "candidate_unverified",
            "source_failure_ids": [failure_id],
            "source_success_ids": [success_id],
            "recovery_source_ids": [failure_id, success_id],
            "stats": {
                "support_count": 1,
                "used_count": 0,
                "success_after_used": 0,
                "failure_after_used": 0,
                "recovered_from_failure": True,
                "recovery_status": "candidate_unverified",
                "source_failure_count": 1,
                "success_prior_count": 1,
            },
        },
    )


def _recovery_limits(
    max_candidates: Optional[int],
    max_success_candidates: Optional[int],
) -> Tuple[int, int, int, int]:
    max_candidates = max_candidates or int(runtime_setting("failure_recovery.max_candidates", 1))
    max_success_candidates = max_success_candidates or int(runtime_setting("failure_recovery.success_candidates", 3))
    max_per_failure = 1
    max_llm_calls = int(runtime_setting("failure_recovery.max_llm_calls", max_candidates))
    return max_candidates, max_success_candidates, max_per_failure, max_llm_calls


def _split_recovery_sources(
    normalized: ExperienceLibrary,
) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, Dict[str, Any]]]]:
    failures = [(exp_id, record) for exp_id, record in normalized.items() if _is_recoverable_candidate(record)]
    successes = [(exp_id, record) for exp_id, record in normalized.items() if _is_success_prior(record)]
    for exp_id, record in successes + failures:
        record["id"] = exp_id
    return failures, successes


def _candidate_success_records(
    failure: Dict[str, Any],
    successes: List[Tuple[str, Dict[str, Any]]],
    max_success_candidates: int,
    existing_pairs: set,
) -> List[Dict[str, Any]]:
    failure_id = str(failure.get("id") or "")
    candidates = _select_success_candidates(failure, successes, max_success_candidates)
    return [
        success
        for success_id, success in candidates
        if (failure_id, success_id) not in existing_pairs
    ]


def _add_recovery_proposals(
    *,
    updated: ExperienceLibrary,
    created: List[str],
    proposals: List[Dict[str, Any]],
    failure_id: str,
    failure: Dict[str, Any],
    success_by_id: Dict[str, Dict[str, Any]],
    existing_pairs: set,
    recovery_counts: Dict[str, int],
    max_candidates: int,
) -> None:
    for proposal in proposals:
        success_id = str(proposal.get("source_success_id") or "").strip()
        if not success_id or (failure_id, success_id) in existing_pairs:
            continue
        recovery_id = _next_experience_id(updated)
        if recovery_id in updated:
            continue
        record = _proposal_to_record(recovery_id, failure_id, failure, proposal, success_by_id)
        if record is None:
            continue
        updated[recovery_id] = record
        existing_pairs.add((failure_id, success_id))
        recovery_counts[failure_id] = recovery_counts.get(failure_id, 0) + 1
        created.append(recovery_id)
        if len(created) >= max_candidates:
            break


def _print_recovery_summary(created: List[str], llm_calls: int, max_llm_calls: int) -> None:
    if created:
        print(f"  [Failure Recovery] LLM added {len(created)} recovery candidate experiences ({llm_calls} call(s)).")
    elif llm_calls >= max_llm_calls:
        print(f"  [Failure Recovery] Skipped remaining failures after {llm_calls} LLM call(s).")


def synthesize_recovery_experiences(
    experiences: ExperienceLibrary,
    *,
    llm: Any = None,
    max_candidates: Optional[int] = None,
    max_success_candidates: Optional[int] = None,
) -> Tuple[ExperienceLibrary, List[str]]:
    """Ask an LLM to convert recoverable failures into candidate success records."""
    normalized = normalize_experience_library(experiences)
    if llm is None:
        return normalized, []

    max_candidates, max_success_candidates, max_per_failure, max_llm_calls = _recovery_limits(
        max_candidates, max_success_candidates
    )
    if max_candidates <= 0:
        return normalized, []
    if max_llm_calls <= 0:
        return normalized, []

    failures, successes = _split_recovery_sources(normalized)
    if not failures or not successes:
        return normalized, []

    synthesizer = _RecoverySynthesizer(llm, max_success_candidates=max_success_candidates)
    existing_pairs = _existing_recovery_pairs(normalized)
    recovery_counts = _recovery_counts_by_failure(normalized)
    success_by_id = {exp_id: record for exp_id, record in successes}
    updated = dict(normalized)
    created: List[str] = []
    llm_calls = 0

    for failure_id, failure in failures:
        if len(created) >= max_candidates or llm_calls >= max_llm_calls:
            break
        if recovery_counts.get(failure_id, 0) >= max_per_failure:
            continue
        candidate_records = _candidate_success_records(
            failure, successes, max_success_candidates, existing_pairs
        )
        if not candidate_records:
            continue

        llm_calls += 1
        _add_recovery_proposals(
            updated=updated,
            created=created,
            proposals=synthesizer.propose_recovery(failure, candidate_records),
            failure_id=failure_id,
            failure=failure,
            success_by_id=success_by_id,
            existing_pairs=existing_pairs,
            recovery_counts=recovery_counts,
            max_candidates=max_candidates,
        )

    _print_recovery_summary(created, llm_calls, max_llm_calls)
    return updated, created
