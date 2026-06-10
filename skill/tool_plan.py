"""Prompt formatting for executable tool plans from selected memory."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from prompts.skill_injection_templates import (
    DEFAULT_TOOL_PLAN_TEMPLATE,
    MANDATORY_TOOL_PLAN_TEMPLATE,
)
from experience.schema import (
    ExperienceLibrary,
    mark_low_information_success,
    normalize_experience_library,
)
from .filters import (
    allow_unvalidated_recovery_inference,
    _skill_usage_rate,
    experience_low_validation_quality,
    experience_unvalidated_recovery,
    recovery_has_train_support,
    skill_low_validation_quality,
)
from .utils import as_callable_tool, callable_tools


GUARDRAIL_TOOL_PLAN_MIN_USES = 2
GUARDRAIL_TOOL_PLAN_MIN_SUCCESS_RATE = 0.70
MIN_VALIDATION_USES_TO_GATE_RECOVERY_TOOL_PLAN = 3
ACTION_SKILL_MIN_VALIDATION_AVERAGE = 0.12
ACTION_SKILL_MIN_SUCCESS_SUPPORT = 2
DIRECT_EXPERIENCE_MIN_VALIDATION_AVG = 0.12
DIRECT_EXPERIENCE_MIN_VALIDATION_PASS = 0.25
DIRECT_EXPERIENCE_TOOL_PLAN_MIN_SUPPORT = 2
ACTION_TOOL_PLAN_MAX_EXPERIENCES = 3
BAD_EXPERIENCE_MAX_VALIDATION_AVG = 0.01
BAD_EXPERIENCE_MAX_VALIDATION_PASS = 0.0


def _record_has_callable_tool_policy(record: Dict[str, Any]) -> bool:
    if not record.get("tool_sequence"):
        return False
    if any(as_callable_tool(tool) for tool in record.get("tool_tags", []) or []):
        return True
    return any(
        isinstance(step, dict) and as_callable_tool(step.get("tool"))
        for step in record.get("tool_sequence", []) or []
    )


def _short_code_template_for_prompt(template: str, max_lines: int = 80, max_chars: int = 6000) -> str:
    """Keep code snippets compact without guessing benchmark-specific constants."""
    lines = [line.rstrip() for line in str(template or "").strip().splitlines()]
    lines = [line for line in lines if line.strip()]
    return "\n".join(lines[:max_lines])[:max_chars].rstrip()


def _stats(item: Dict[str, Any]) -> Dict[str, Any]:
    stats = item.get("stats")
    return stats if isinstance(stats, dict) else {}


def _support_counts(item: Dict[str, Any]) -> Dict[str, Any]:
    support = item.get("support_counts")
    return support if isinstance(support, dict) else {}


def _validation_feedback(
    item: Dict[str, Any],
    quality_fallback: float = 0.0,
    use_item_quality: bool = True,
) -> Tuple[int, float, float]:
    stats = _stats(item)
    fallback_quality = item.get("validation_quality", quality_fallback) if use_item_quality else quality_fallback
    return (
        int(stats.get("validation_used_count", 0)),
        float(stats.get("validation_average_score", fallback_quality)),
        float(stats.get("validation_pass_rate", 0.0)),
    )


def _failure_guardrail_tool_plan_allowed(skill: Dict[str, Any]) -> bool:
    metadata = skill.get("metadata", {}) if isinstance(skill, dict) else {}
    if metadata.get("skill_type") != "failure_guardrail":
        return False
    used_count, usage_rate = _skill_usage_rate(metadata)
    if usage_rate is None:
        return False
    min_uses = GUARDRAIL_TOOL_PLAN_MIN_USES
    min_rate = GUARDRAIL_TOOL_PLAN_MIN_SUCCESS_RATE
    return used_count >= min_uses and usage_rate >= min_rate


def _action_skill_tool_plan_allowed(skill: Dict[str, Any]) -> bool:
    """Gate mandatory execution to skills with enough evidence to drive action."""
    metadata = skill.get("metadata", {}) if isinstance(skill, dict) else {}
    if metadata.get("skill_type") == "failure_guardrail":
        return False
    if metadata.get("recovery_status") == "rejected":
        return False
    if skill_low_validation_quality(metadata):
        return False
    if not metadata.get("requires_tool_plan", True):
        return False

    val_used, val_quality, _ = _validation_feedback(metadata)
    if val_used:
        if metadata.get("skill_type") == "recovery_action_skill":
            min_val_uses = MIN_VALIDATION_USES_TO_GATE_RECOVERY_TOOL_PLAN
            if val_used < min_val_uses:
                return True
        allowed = val_quality >= ACTION_SKILL_MIN_VALIDATION_AVERAGE
        if metadata.get("skill_type") == "recovery_action_skill":
            if allowed:
                metadata["recovery_status"] = "validated"
            return allowed
        return allowed

    support = _support_counts(metadata)
    success_support = int(support.get("success", 0))
    failure_support = int(support.get("failure", 0))
    recovery_support = int(support.get("recovery", 0))
    success_prior_support = int(support.get("success_prior", 0))
    supervised_support = int(support.get("supervised_correction", 0))

    if metadata.get("skill_type") == "recovery_action_skill":
        return (
            recovery_support > 0
            and (success_prior_support > 0 or supervised_support > 0)
        )

    min_success_support = ACTION_SKILL_MIN_SUCCESS_SUPPORT
    return success_support >= min_success_support and success_support >= failure_support


def _direct_experience_tool_plan_allowed(record: Dict[str, Any]) -> bool:
    """Allow standalone experience policies to drive action only with evidence.

    Retrieved experiences are useful context, but a single unvalidated rollout
    should not become a mandatory executable plan. This gate keeps direct
    experience plans for memories with validation feedback or repeated support.
    """
    if record.get("kind") != "success":
        return False
    if (
        experience_unvalidated_recovery(record)
        and not allow_unvalidated_recovery_inference()
        and not recovery_has_train_support(record)
    ):
        return False
    if experience_low_validation_quality(record):
        return False

    stats = _stats(record)
    val_used, val_avg, val_pass = _validation_feedback(record, use_item_quality=False)
    if (
        val_used <= 0
        and stats.get("supervised_correction")
        and experience_unvalidated_recovery(record)
    ):
        return True

    if val_used:
        min_avg = DIRECT_EXPERIENCE_MIN_VALIDATION_AVG
        min_pass = DIRECT_EXPERIENCE_MIN_VALIDATION_PASS
        return val_avg >= min_avg or val_pass >= min_pass

    return int(stats.get("support_count", 1)) >= DIRECT_EXPERIENCE_TOOL_PLAN_MIN_SUPPORT


def _record_quality_rank_key(record: Dict[str, Any]) -> Tuple[Any, ...]:
    stats = _stats(record)
    val_used, val_avg, val_pass = _validation_feedback(record, use_item_quality=False)
    return (
        val_used > 0 and (val_avg > 0.0 or val_pass > 0.0),
        bool(stats.get("supervised_correction")),
        int(stats.get("support_count", 1) or 1) >= DIRECT_EXPERIENCE_TOOL_PLAN_MIN_SUPPORT,
        val_avg,
        val_pass,
        bool(stats.get("recovered_from_failure")),
    )


def _select_action_experience_ids(exp_ids: List[str], normalized: Dict[str, Dict[str, Any]]) -> set:
    max_ids = ACTION_TOOL_PLAN_MAX_EXPERIENCES
    if max_ids <= 0:
        return set(exp_ids)
    unique = [exp_id for exp_id in dict.fromkeys(exp_ids) if exp_id in normalized]
    ranked = sorted(unique, key=lambda exp_id: _record_quality_rank_key(normalized[exp_id]), reverse=True)
    protected = [
        exp_id for exp_id in unique
        if (normalized[exp_id].get("stats") or {}).get("supervised_correction")
        and exp_id not in ranked[:max_ids]
    ]
    selected = list(dict.fromkeys(ranked[:max_ids] + protected[:1]))
    return set(selected)


def _validation_failed_bad_threshold(validation_used: int, validation_avg: float, validation_pass: float) -> bool:
    return (
        validation_used > 0
        and validation_avg <= BAD_EXPERIENCE_MAX_VALIDATION_AVG
        and validation_pass <= BAD_EXPERIENCE_MAX_VALIDATION_PASS
    )


def _normalized_experience_records(selected_experiences: ExperienceLibrary) -> Dict[str, Dict[str, Any]]:
    return {
        exp_id: mark_low_information_success(record)
        for exp_id, record in normalize_experience_library(selected_experiences).items()
    }


def _selected_tool_plan_skill_ids(selected_skills: List[Dict[str, Any]]) -> Tuple[set, set, set]:
    action_skill_ids = {
        str(skill.get("skill_id"))
        for skill in selected_skills or []
        if _action_skill_tool_plan_allowed(skill)
    }
    guardrail_tool_skill_ids = {
        str(skill.get("skill_id"))
        for skill in selected_skills or []
        if _failure_guardrail_tool_plan_allowed(skill)
        and any(as_callable_tool(tool) for tool in skill.get("metadata", {}).get("tool_tags", []) or [])
    }
    return action_skill_ids, guardrail_tool_skill_ids, action_skill_ids | guardrail_tool_skill_ids


def _linked_experience_ids_for_skills(selected_skills: List[Dict[str, Any]], skill_ids: set) -> List[str]:
    return [
        str(exp_id)
        for skill in selected_skills or []
        if str(skill.get("skill_id")) in skill_ids
        for exp_id in skill.get("metadata", {}).get("linked_experience_ids", []) or []
    ]


def _guardrail_experience_ids(selected_skills: List[Dict[str, Any]], guardrail_tool_skill_ids: set) -> set:
    return {
        str(exp_id)
        for skill in selected_skills or []
        if str(skill.get("skill_id")) in guardrail_tool_skill_ids
        if skill.get("metadata", {}).get("skill_type") == "failure_guardrail"
        for exp_id in skill.get("metadata", {}).get("linked_experience_ids", []) or []
    }


def _add_unique(items: List[str], line: str) -> None:
    if line not in items:
        items.append(line)


def _collect_skill_tools(selected_skills: List[Dict[str, Any]], tool_plan_skill_ids: set) -> List[str]:
    tools: List[str] = []
    for skill in selected_skills or []:
        if str(skill.get("skill_id")) not in tool_plan_skill_ids:
            continue
        for tool in skill.get("metadata", {}).get("tool_tags", []) or []:
            tool = as_callable_tool(tool)
            if tool and tool not in tools:
                tools.append(tool)
    return tools


def _record_recovery_blocked(record: Dict[str, Any]) -> bool:
    return (
        experience_unvalidated_recovery(record)
        and not allow_unvalidated_recovery_inference()
        and not recovery_has_train_support(record)
    )


def _low_information_success(record: Dict[str, Any]) -> bool:
    return (
        record.get("kind") == "success"
        and _stats(record).get("low_information_success")
    )


def _direct_success_policy_allowed(
    record: Dict[str, Any],
    selected_skills: List[Dict[str, Any]],
    tool_plan_skill_ids: set,
    recovery_status: Any,
    validation_failed: bool,
    has_direct_tool_policy: bool,
) -> bool:
    return (
        not selected_skills
        and not tool_plan_skill_ids
        and recovery_status != "rejected"
        and not validation_failed
        and has_direct_tool_policy
        and _direct_experience_tool_plan_allowed(record)
    )


def _record_action_flags(
    exp_id: str,
    record: Dict[str, Any],
    selected_skills: List[Dict[str, Any]],
    tool_plan_skill_ids: set,
    action_exp_ids: set,
    guardrail_exp_ids: set,
) -> Tuple[bool, bool]:
    has_direct_tool_policy = _record_has_callable_tool_policy(record)
    stats = _stats(record)
    recovery_status = record.get("recovery_status") or stats.get("recovery_status")
    validation_used, validation_avg, validation_pass = _validation_feedback(record, use_item_quality=False)
    validation_failed = _validation_failed_bad_threshold(validation_used, validation_avg, validation_pass)
    direct_success_policy = _direct_success_policy_allowed(
        record, selected_skills, tool_plan_skill_ids, recovery_status, validation_failed, has_direct_tool_policy
    )
    should_drive_action = (exp_id in action_exp_ids and record.get("kind") == "success") or (
        exp_id in guardrail_exp_ids and has_direct_tool_policy
    ) or direct_success_policy
    if _low_information_success(record) or experience_low_validation_quality(record):
        should_drive_action = False
    include_supporting_checks = should_drive_action or (
        record.get("kind") == "failure" and exp_id in guardrail_exp_ids
    )
    return should_drive_action, include_supporting_checks


def _append_record_tools(tools: List[str], record: Dict[str, Any], should_drive_action: bool) -> None:
    if not should_drive_action:
        return
    for tool in record.get("tool_tags", []) or []:
        tool = as_callable_tool(tool)
        if tool and tool not in tools:
            tools.append(tool)


def _append_record_steps(
    steps: List[str],
    exp_id: str,
    record: Dict[str, Any],
    should_drive_action: bool,
    max_steps: int,
) -> None:
    if not should_drive_action:
        return
    for step in record.get("tool_sequence", []) or []:
        if not isinstance(step, dict):
            continue
        tool = as_callable_tool(step.get("tool"))
        if not tool:
            continue
        purpose = str(step.get("purpose") or "run the tool-backed check").strip()
        expected = str(step.get("expected_output") or "").strip()
        line = f"{len(steps) + 1}. `{tool}`: {purpose}"
        if expected:
            line += f" -> expected output: {expected}"
        line += f" [{exp_id}]"
        _add_unique(steps, line)
        if len(steps) >= max_steps:
            break


def _append_record_supporting_text(
    validations: List[str],
    fallbacks: List[str],
    exp_id: str,
    record: Dict[str, Any],
    include_supporting_checks: bool,
) -> None:
    if not include_supporting_checks:
        return
    for check in record.get("validation_checks", []) or []:
        _add_unique(validations, f"- {check} [{exp_id}]")
    for fallback in record.get("fallback_plan", []) or []:
        _add_unique(fallbacks, f"- {fallback} [{exp_id}]")


def _append_record_code_template(
    code_templates: List[str],
    exp_id: str,
    record: Dict[str, Any],
    should_drive_action: bool,
) -> None:
    if len(code_templates) >= 2:
        return
    if not should_drive_action or record.get("kind") != "success" or not record.get("code_template"):
        return
    code_template = _short_code_template_for_prompt(record["code_template"])
    if code_template:
        code_templates.append(f"# From {exp_id}\n{code_template}")


def _collect_experience_tool_plan_parts(
    normalized: Dict[str, Dict[str, Any]],
    selected_skills: List[Dict[str, Any]],
    tool_plan_skill_ids: set,
    action_exp_ids: set,
    guardrail_exp_ids: set,
    max_steps: int,
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    tools: List[str] = []
    steps: List[str] = []
    validations: List[str] = []
    fallbacks: List[str] = []
    code_templates: List[str] = []

    for exp_id, record in normalized.items():
        if _record_recovery_blocked(record):
            continue
        should_drive_action, include_supporting_checks = _record_action_flags(
            exp_id, record, selected_skills, tool_plan_skill_ids, action_exp_ids, guardrail_exp_ids
        )
        _append_record_tools(tools, record, should_drive_action)
        _append_record_steps(steps, exp_id, record, should_drive_action, max_steps)
        _append_record_supporting_text(validations, fallbacks, exp_id, record, include_supporting_checks)
        _append_record_code_template(code_templates, exp_id, record, should_drive_action)

    return tools, steps, validations, fallbacks, code_templates


def _default_tool_plan_steps(tools: List[str]) -> List[str]:
    return [
        f"1. `{tools[0]}`: derive the tool input from the current image/problem, execute the check, and inspect the visible result before answering."
    ]


def _code_template_text(code_templates: List[str]) -> str:
    if not code_templates:
        return ""
    blocks = [f"```python\n{template}\n```" for template in code_templates]
    return "\n\nReusable code template(s):\n" + "\n".join(blocks)


def _render_tool_plan_block(
    tools: List[str],
    steps: List[str],
    validations: List[str],
    fallbacks: List[str],
    code_templates: List[str],
    max_steps: int,
) -> str:
    if not steps:
        steps = _default_tool_plan_steps(tools)
    if not validations:
        validations = ["- Inspect a visible intermediate result; do not rely only on a printed object handle or unverified estimate."]
    if not fallbacks:
        fallbacks = ["- If the first tool output is empty, noisy, or inconsistent, adjust the crop/threshold/input and rerun one tool-backed check."]

    return MANDATORY_TOOL_PLAN_TEMPLATE.format(
        tool_text=", ".join(f"`{tool}`" for tool in tools),
        steps="\n".join(steps[:max_steps]),
        validations="\n".join(validations[:6]),
        fallbacks="\n".join(fallbacks[:4]),
        code_templates=_code_template_text(code_templates),
    ).strip()


def format_executable_tool_plan_for_prompt(
    selected_skills: List[Dict[str, Any]],
    selected_experiences: ExperienceLibrary,
    max_steps: int = 6,
) -> Tuple[str, List[str]]:
    """Create a strict, compact tool-plan block from selected skills and experiences."""
    normalized = _normalized_experience_records(selected_experiences)
    _, guardrail_tool_skill_ids, tool_plan_skill_ids = _selected_tool_plan_skill_ids(selected_skills)
    action_exp_ordered = _linked_experience_ids_for_skills(selected_skills, tool_plan_skill_ids)
    action_exp_ids = _select_action_experience_ids(action_exp_ordered, normalized)
    guardrail_exp_ids = _guardrail_experience_ids(selected_skills, guardrail_tool_skill_ids)
    tools = _collect_skill_tools(selected_skills, tool_plan_skill_ids)
    exp_tools, steps, validations, fallbacks, code_templates = _collect_experience_tool_plan_parts(
        normalized,
        selected_skills,
        tool_plan_skill_ids,
        action_exp_ids,
        guardrail_exp_ids,
        max_steps,
    )
    for tool in exp_tools:
        if tool not in tools:
            tools.append(tool)

    if not tools:
        return "", []
    return _render_tool_plan_block(tools, steps, validations, fallbacks, code_templates, max_steps), tools


def format_default_tool_plan_for_prompt() -> Tuple[str, List[str]]:
    """Create a generic one-tool minimum plan when no memory policy is available."""
    tools = callable_tools()
    tool_text = ", ".join(f"`{tool}`" for tool in tools)
    search_tools = [tool for tool in ("web_search", "image_search", "visit") if tool in tools]
    steps = []
    if "code_interpreter" in tools and search_tools:
        steps.append(
            "1. Call at least one enabled tool before answering: use `code_interpreter` "
            "for visible image/layout/counting/computation evidence, or use "
            + "/".join(f"`{tool}`" for tool in search_tools)
            + " when the answer depends on external identity, source, or factual lookup."
        )
    elif "code_interpreter" in tools:
        steps.append(
            "1. `code_interpreter`: inspect, crop, compute, or visualize one decision-relevant "
            "piece of evidence from the current image/problem before answering."
        )
    elif search_tools:
        steps.append(
            "1. Call the most relevant search or visit tool to verify the visual entity, source, "
            "or factual clue before answering."
        )
    else:
        steps.append(
            f"1. `{tools[0]}`: run one focused tool-backed check from the current image/problem before answering."
        )

    block = DEFAULT_TOOL_PLAN_TEMPLATE.format(
        tool_text=tool_text,
        steps="\n".join(steps),
    )
    return block.strip(), tools

