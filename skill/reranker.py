"""LLM reranker and formatting for selected memory."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from experience.schema import (
    ExperienceLibrary,
    format_experience_for_prompt,
    normalize_experience_library,
)
from .config import SkillSettings
from .filters import (
    allow_unvalidated_recovery_inference,
    experience_failed_validation,
    experience_low_validation_quality,
    experience_unvalidated_recovery,
    recovery_has_train_support,
    selection_gate_allows_skill,
)
from utils.json_utils import parse_llm_json_object
from .utils import load_memory_prompt, truncate_text


RECOVERY_MAX_UNVERIFIED_SELECTED_SKILLS = 1


def _settings() -> SkillSettings:
    return SkillSettings.from_runtime()


def _format_skill_candidate_for_rerank(skill: Dict[str, Any], max_chars: int = 1800) -> str:
    metadata = skill.get("metadata", {}) if isinstance(skill, dict) else {}
    skill_id = skill.get("skill_id", "") if isinstance(skill, dict) else ""
    stats = metadata.get("stats") if isinstance(metadata.get("stats"), dict) else {}
    support_counts = metadata.get("support_counts") if isinstance(metadata.get("support_counts"), dict) else {}
    fields = [
        f"[{skill_id}] score={float(skill.get('score', 0.0) or 0.0):.3f}",
        f"skill_type: {metadata.get('skill_type', '')}",
        f"recovery_status: {metadata.get('recovery_status', '')}",
        f"pending_validation_revision: {metadata.get('pending_validation_revision', '')}",
        f"requires_tool_plan: {metadata.get('requires_tool_plan', '')}",
        f"validation_quality: {metadata.get('validation_quality', stats.get('validation_average_score', ''))}",
        f"validation_used_count: {stats.get('validation_used_count', '')}",
        f"support_counts: success={support_counts.get('success', '')}, failure={support_counts.get('failure', '')}",
        f"description: {metadata.get('description', '')}",
        f"task_tags: {', '.join(metadata.get('task_tags', []) or [])}",
        f"tool_tags: {', '.join(metadata.get('tool_tags', []) or [])}",
        f"linked_experience_ids: {', '.join(metadata.get('linked_experience_ids', []) or [])}",
        "content:",
        truncate_text(skill.get("content", ""), max_chars),
    ]
    return "\n".join(fields)


def _format_experience_candidate_for_rerank(exp_id: str, record: Dict[str, Any], max_chars: int = 1200) -> str:
    text = format_experience_for_prompt(exp_id, record)
    return f"[{exp_id}]\n{truncate_text(text, max_chars)}"


def _experience_can_drive_action(record: Dict[str, Any]) -> bool:
    """Prefer positive executable priors over failure-only warnings."""
    if not isinstance(record, dict) or record.get("kind") != "success":
        return False
    if experience_failed_validation(record) or experience_low_validation_quality(record):
        return False
    if (
        experience_unvalidated_recovery(record)
        and not allow_unvalidated_recovery_inference()
        and not recovery_has_train_support(record)
    ):
        return False
    tools = {str(tool).strip() for tool in record.get("tool_tags", []) or []}
    return bool(record.get("tool_sequence")) and bool(tools - {"", "reasoning"})


def _ensure_success_prior(
    exp_by_id: Dict[str, Dict[str, Any]],
    exp_ids: List[str],
    max_experiences: int,
) -> List[str]:
    """Keep at least one executable success prior when candidates contain one.

    The LLM reranker can over-select failure memories because they mention the
    exact error mode. For inference, a failure warning without a positive tool
    policy often leaves the agent with no executable prior. This helper keeps
    the highest-ranked success candidate while preserving the LLM order for the
    rest of the selection.
    """
    if max_experiences <= 0:
        return []
    if any(_experience_can_drive_action(exp_by_id.get(exp_id, {})) for exp_id in exp_ids):
        return exp_ids[:max_experiences]

    for exp_id, record in exp_by_id.items():
        if _experience_can_drive_action(record):
            merged = [exp_id] + [item for item in exp_ids if item != exp_id]
            return merged[:max_experiences]
    return exp_ids[:max_experiences]


def _ensure_linked_experiences_for_selected_skills(
    selected_skills: List[Dict[str, Any]],
    exp_by_id: Dict[str, Dict[str, Any]],
    exp_ids: List[str],
    max_experiences: int,
) -> List[str]:
    """Keep executable evidence attached to selected skills.

    The reranker is allowed to select a skill without repeating every linked
    experience id. Downstream tool-plan construction still needs those linked
    records, so selected skill evidence is treated as part of the skill rather
    than as optional direct memory.
    """
    if max_experiences <= 0:
        return []
    linked_ids: List[str] = []
    for skill in selected_skills:
        metadata = skill.get("metadata", {}) if isinstance(skill, dict) else {}
        for exp_id in metadata.get("linked_experience_ids", []) or []:
            exp_id = str(exp_id)
            if exp_id in exp_by_id and exp_id not in linked_ids:
                linked_ids.append(exp_id)
    merged = linked_ids + [exp_id for exp_id in exp_ids if exp_id not in linked_ids]
    return merged[:max_experiences]


def _limit_unverified_recovery_skills(
    skill_ids: List[str],
    skill_by_id: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, str]]]:
    """Keep unverified recovery probes compact after LLM reranking."""
    max_unverified = RECOVERY_MAX_UNVERIFIED_SELECTED_SKILLS
    if max_unverified < 0:
        return skill_ids, []
    kept: List[str] = []
    dropped: List[Dict[str, str]] = []
    unverified_seen = 0
    for skill_id in skill_ids:
        metadata = skill_by_id.get(skill_id, {}).get("metadata", {})
        is_unverified = (
            metadata.get("skill_type") == "recovery_action_skill"
            and str(metadata.get("recovery_status") or "") == "candidate_unverified"
        )
        if is_unverified:
            if unverified_seen >= max_unverified:
                dropped.append({"skill_id": skill_id, "reason": "too_many_unverified_recovery_skills"})
                continue
            unverified_seen += 1
        kept.append(skill_id)
    return kept, dropped


def _candidate_maps(
    candidate_skills: List[Dict[str, Any]],
    candidate_experiences: ExperienceLibrary,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    skill_by_id = {
        str(skill.get("skill_id")): skill
        for skill in candidate_skills
        if isinstance(skill, dict) and skill.get("skill_id")
    }
    normalized_experiences = normalize_experience_library(candidate_experiences)
    exp_by_id = {
        str(exp_id): record
        for exp_id, record in normalized_experiences.items()
        if not experience_failed_validation(record)
        and not experience_low_validation_quality(record)
    }
    return skill_by_id, exp_by_id


def _fallback_exp_ids(exp_by_id: Dict[str, Dict[str, Any]], max_experiences: int) -> List[str]:
    return _ensure_success_prior(exp_by_id, list(exp_by_id.keys())[:max_experiences], max_experiences)


def _fallback_info(
    question_id: Optional[str],
    skill_by_id: Dict[str, Dict[str, Any]],
    exp_by_id: Dict[str, Dict[str, Any]],
    fallback_skills: List[Dict[str, Any]],
    fallback_experiences: ExperienceLibrary,
    query_metadata_tags: Optional[Iterable[str]],
) -> Dict[str, Any]:
    return {
        "rerank_used": False,
        "question_id": question_id,
        "candidate_skill_ids": list(skill_by_id.keys()),
        "candidate_experience_ids": list(exp_by_id.keys()),
        "selected_skill_ids": [skill.get("skill_id") for skill in fallback_skills],
        "selected_experience_ids": list(fallback_experiences.keys()),
        "query_metadata_tags": list(query_metadata_tags or []),
        "fallback": False,
    }


def _gate_selected_skill_ids(
    skill_ids: List[str],
    skill_by_id: Dict[str, Dict[str, Any]],
    query_metadata_tags: Optional[Iterable[str]],
) -> Tuple[List[str], List[Dict[str, str]]]:
    gated_skill_ids = []
    dropped_skill_ids: List[Dict[str, str]] = []
    for skill_id in skill_ids:
        allowed, reason = selection_gate_allows_skill(skill_by_id[skill_id], query_metadata_tags)
        if allowed:
            gated_skill_ids.append(skill_id)
        else:
            dropped_skill_ids.append({"skill_id": skill_id, "reason": reason})
    gated_skill_ids, extra_drops = _limit_unverified_recovery_skills(gated_skill_ids, skill_by_id)
    dropped_skill_ids.extend(extra_drops)
    return gated_skill_ids, dropped_skill_ids


def _gate_fallback_skills(
    fallback_skills: List[Dict[str, Any]],
    query_metadata_tags: Optional[Iterable[str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    gated_fallback_skills = []
    dropped_skill_ids: List[Dict[str, str]] = []
    for skill in fallback_skills:
        allowed, reason = selection_gate_allows_skill(skill, query_metadata_tags)
        if allowed:
            gated_fallback_skills.append(skill)
        else:
            dropped_skill_ids.append({"skill_id": str(skill.get("skill_id") or ""), "reason": reason})
    return gated_fallback_skills, dropped_skill_ids


def _fallback_selection(
    fallback_skills: List[Dict[str, Any]],
    exp_by_id: Dict[str, Dict[str, Any]],
    fallback_exp_ids: List[str],
    max_experiences: int,
    query_metadata_tags: Optional[Iterable[str]],
) -> Tuple[List[Dict[str, Any]], ExperienceLibrary, List[Dict[str, str]]]:
    gated_fallback_skills, dropped_skill_ids = _gate_fallback_skills(fallback_skills, query_metadata_tags)
    exp_ids = _ensure_linked_experiences_for_selected_skills(
        gated_fallback_skills,
        exp_by_id,
        fallback_exp_ids,
        max_experiences,
    )
    fallback_experiences = {exp_id: exp_by_id[exp_id] for exp_id in exp_ids if exp_id in exp_by_id}
    return gated_fallback_skills, fallback_experiences, dropped_skill_ids


def _update_fallback_info(
    info: Dict[str, Any],
    selected_skills: List[Dict[str, Any]],
    selected_experiences: ExperienceLibrary,
    dropped_skill_ids: List[Dict[str, str]],
    error: str,
) -> None:
    info.update({
        "fallback": True,
        "error": error,
        "selected_skill_ids": [skill.get("skill_id") for skill in selected_skills],
        "selected_experience_ids": list(selected_experiences.keys()),
        "dropped_skill_ids": dropped_skill_ids,
    })


def _rerank_prompt(
    query_text: str,
    candidate_skills: List[Dict[str, Any]],
    skill_by_id: Dict[str, Dict[str, Any]],
    exp_by_id: Dict[str, Dict[str, Any]],
    max_skills: int,
    max_experiences: int,
) -> str:
    skill_text = "\n\n".join(
        _format_skill_candidate_for_rerank(skill)
        for skill in candidate_skills
        if skill.get("skill_id") in skill_by_id
    ) or "(none)"
    exp_text = "\n\n".join(
        _format_experience_candidate_for_rerank(exp_id, record)
        for exp_id, record in exp_by_id.items()
    ) or "(none)"
    return load_memory_prompt("memory_rerank").format(
        query=query_text,
        skills=skill_text,
        experiences=exp_text,
        max_skills=max_skills,
        max_experiences=max_experiences,
    )


def _parse_rerank_response(response: str) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
    parsed = parse_llm_json_object(response)
    if "skill_ids" not in parsed or "experience_ids" not in parsed:
        raise ValueError("rerank response missing skill_ids or experience_ids")
    raw_skill_ids = parsed.get("skill_ids", [])
    raw_exp_ids = parsed.get("experience_ids", [])
    if not isinstance(raw_skill_ids, list) or not isinstance(raw_exp_ids, list):
        raise ValueError("rerank response skill_ids/experience_ids must be lists")
    return raw_skill_ids, raw_exp_ids, parsed


def _filter_llm_ids(
    raw_skill_ids: List[Any],
    raw_exp_ids: List[Any],
    skill_by_id: Dict[str, Dict[str, Any]],
    exp_by_id: Dict[str, Dict[str, Any]],
    max_skills: int,
    max_experiences: int,
) -> Tuple[List[str], List[str]]:
    skill_ids = [
        str(skill_id)
        for skill_id in raw_skill_ids
        if str(skill_id) in skill_by_id
    ][:max_skills]
    exp_ids = [
        str(exp_id)
        for exp_id in raw_exp_ids
        if str(exp_id) in exp_by_id
    ][:max_experiences]
    exp_ids = _ensure_success_prior(exp_by_id, exp_ids, max_experiences)
    return skill_ids, exp_ids


def _write_rerank_debug(debug_dir: Optional[str], prompt: str, response: str, info: Dict[str, Any]) -> None:
    if not debug_dir:
        return
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    Path(debug_dir, "tt_memory_rerank.json").write_text(
        json.dumps({"prompt": prompt, "response": response, "info": info}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _initial_rerank_state(
    candidate_skills: List[Dict[str, Any]],
    candidate_experiences: ExperienceLibrary,
    max_skills: int,
    max_experiences: int,
    question_id: Optional[str],
    query_metadata_tags: Optional[Iterable[str]],
):
    skill_by_id, exp_by_id = _candidate_maps(candidate_skills, candidate_experiences)
    fallback_skills = candidate_skills[:max_skills]
    fallback_exp_ids = _fallback_exp_ids(exp_by_id, max_experiences)
    fallback_experiences = {exp_id: exp_by_id[exp_id] for exp_id in fallback_exp_ids if exp_id in exp_by_id}
    info = _fallback_info(
        question_id, skill_by_id, exp_by_id, fallback_skills, fallback_experiences, query_metadata_tags
    )
    return skill_by_id, exp_by_id, fallback_skills, fallback_exp_ids, info


def _fallback_rerank_result(
    info: Dict[str, Any],
    fallback_skills: List[Dict[str, Any]],
    exp_by_id: Dict[str, Dict[str, Any]],
    fallback_exp_ids: List[str],
    max_experiences: int,
    query_metadata_tags: Optional[Iterable[str]],
    error: str,
):
    selected_skills, selected_experiences, dropped_skill_ids = _fallback_selection(
        fallback_skills,
        exp_by_id,
        fallback_exp_ids,
        max_experiences,
        query_metadata_tags,
    )
    _update_fallback_info(info, selected_skills, selected_experiences, dropped_skill_ids, error)
    return selected_skills, selected_experiences, info


def _select_from_rerank_response(
    response: str,
    skill_by_id: Dict[str, Dict[str, Any]],
    exp_by_id: Dict[str, Dict[str, Any]],
    max_skills: int,
    max_experiences: int,
    query_metadata_tags: Optional[Iterable[str]],
):
    raw_skill_ids, raw_exp_ids, parsed = _parse_rerank_response(response)
    skill_ids, exp_ids = _filter_llm_ids(
        raw_skill_ids, raw_exp_ids, skill_by_id, exp_by_id, max_skills, max_experiences
    )
    dropped_skill_ids = []
    if skill_ids:
        skill_ids, dropped_skill_ids = _gate_selected_skill_ids(skill_ids, skill_by_id, query_metadata_tags)

    selected_skills = [skill_by_id[skill_id] for skill_id in skill_ids]
    exp_ids = _ensure_linked_experiences_for_selected_skills(
        selected_skills,
        exp_by_id,
        exp_ids,
        max_experiences,
    )
    selected_experiences = {exp_id: exp_by_id[exp_id] for exp_id in exp_ids}
    return selected_skills, selected_experiences, skill_ids, exp_ids, dropped_skill_ids, parsed


def rerank_memory_with_llm(
    query_text: str,
    candidate_skills: List[Dict[str, Any]],
    candidate_experiences: ExperienceLibrary,
    llm: Any,
    max_skills: int = 2,
    max_experiences: int = 8,
    debug_dir: Optional[str] = None,
    question_id: Optional[str] = None,
    query_metadata_tags: Optional[Iterable[str]] = None,
) -> Tuple[List[Dict[str, Any]], ExperienceLibrary, Dict[str, Any]]:
    """Use an LLM to select useful skills/experiences from embedding candidates."""
    max_skills = max(0, int(max_skills or 0))
    max_experiences = max(0, int(max_experiences or 0))
    skill_by_id, exp_by_id, fallback_skills, fallback_exp_ids, info = _initial_rerank_state(
        candidate_skills,
        candidate_experiences,
        max_skills,
        max_experiences,
        question_id,
        query_metadata_tags,
    )

    if not skill_by_id and not exp_by_id:
        return [], {}, info
    if llm is None:
        return _fallback_rerank_result(
            info, fallback_skills, exp_by_id, fallback_exp_ids, max_experiences, query_metadata_tags, "llm_not_available"
        )

    prompt = _rerank_prompt(
        query_text, candidate_skills, skill_by_id, exp_by_id, max_skills, max_experiences
    )

    try:
        settings = _settings()
        response = llm.chat(
            prompt,
            max_tokens=settings.rerank_max_tokens,
            temperature=settings.rerank_temperature,
            model_name=getattr(llm, "critique_model_name", None),
        )
        selected_skills, selected_experiences, skill_ids, exp_ids, dropped_skill_ids, parsed = (
            _select_from_rerank_response(
                response,
                skill_by_id,
                exp_by_id,
                max_skills,
                max_experiences,
                query_metadata_tags,
            )
        )
        info.update({
            "rerank_used": True,
            "selected_skill_ids": skill_ids,
            "selected_experience_ids": exp_ids,
            "notes": parsed.get("notes", ""),
            "dropped_skill_ids": dropped_skill_ids,
            "fallback": False,
        })
        _write_rerank_debug(debug_dir, prompt, response, info)
        return selected_skills, selected_experiences, info
    except Exception as e:
        print(f"  [Memory Rerank] Warning: LLM rerank failed, using embedding order: {e}")
        return _fallback_rerank_result(
            info, fallback_skills, exp_by_id, fallback_exp_ids, max_experiences, query_metadata_tags, str(e)
        )


def collect_linked_experiences(selected_skills: List[Dict[str, Any]], experiences: ExperienceLibrary) -> ExperienceLibrary:
    normalized = normalize_experience_library(experiences)
    linked = {}
    for skill in selected_skills:
        for exp_id in skill.get("metadata", {}).get("linked_experience_ids", []):
            if exp_id in normalized:
                linked[exp_id] = normalized[exp_id]
    return linked


def _strip_code_blocks_for_skill_prompt(content: str) -> str:
    """Keep skill prompts conceptual; executable code comes from tool plans."""
    replacement = (
        "```text\n"
        "[Executable code omitted here. Use the mandatory tool plan and linked "
        "experience code templates for runnable details.]\n"
        "```"
    )
    return re.sub(r"```(?:python|py)?\s*\n.*?```", replacement, content or "", flags=re.DOTALL | re.IGNORECASE)


def format_selected_skills_for_prompt(selected_skills: List[Dict[str, Any]]) -> str:
    chunks = []
    for skill in selected_skills:
        content = _strip_code_blocks_for_skill_prompt(skill.get("content", ""))
        chunks.append(f"<!-- {skill['skill_id']} score={skill['score']:.3f} -->\n{content}")
    return "\n\n".join(chunks)
