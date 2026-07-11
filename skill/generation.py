"""Generate modular skill markdown from clustered experiences."""

from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from prompts.skill_generation_templates import (
    FAILURE_GUARDRAIL_SKILL_MARKDOWN_TEMPLATE,
    FALLBACK_SKILL_MARKDOWN_TEMPLATE,
    GENERATE_CLUSTER_SKILL_PROMPT,
)
from experience.schema import (
    ExperienceLibrary,
    format_experience_for_prompt,
    normalize_experience_library,
    normalize_experience_record,
)
from configs.runtime import runtime_setting
from .config import SkillSettings
from .failure_recovery import synthesize_recovery_experiences
from .filters import experience_failed_validation, experience_low_validation_quality
from .index import (
    SkillIndex,
    load_skill_index,
    prune_stale_skills,
    save_skill_index,
    skills_dir,
)
from .utils import (
    cluster_task_tags,
    cosine_similarity_matrix,
    skill_name_token,
    slugify,
)


def _settings() -> SkillSettings:
    return SkillSettings.from_env()


_PREVIOUS_SKILL_METADATA_KEYS = (
    "stats",
    "needs_refine",
    "validation_quality",
    "disabled",
    "disabled_reason",
    "recovery_status",
    "requires_validation",
    "source_failure_experience_ids",
    "source_success_experience_ids",
)

DEFAULT_VALIDATION_LINE = "- Inspect at least one visible intermediate result before finalizing."
DEFAULT_FALLBACK_LINE = (
    "- If the first tool result is empty or inconsistent, revise the crop/threshold/input "
    "and rerun the tool once before answering."
)
MAX_TASK_TAGS = 10
MAX_TOOL_TAGS = 4
SKILL_ID_REUSE_MIN_LINK_OVERLAP = 0.5
SKILL_FEEDBACK_REUSE_MIN_LINK_OVERLAP = 0.34
VALIDATION_MIN_USES_TO_DISABLE = 3
VALIDATION_DISABLE_AVERAGE_THRESHOLD = 0.01
SKILL_GENERATION_CACHE_VERSION = 1
VALIDATION_DISABLE_PASS_THRESHOLD = 0.0
LOW_QUALITY_SKILL_MIN_VALIDATION_AVG = 0.12
LOW_QUALITY_SKILL_MAX_PASS_RATE = 0.25
RECOVERY_PROMOTE_AVERAGE_THRESHOLD = 0.25
RECOVERY_PROMOTE_PASS_THRESHOLD = 0.50
RECOVERY_REJECT_MIN_VALIDATION_USES = 3
RECOVERY_REJECT_AVERAGE_THRESHOLD = 0.01
RECOVERY_REJECT_PASS_THRESHOLD = 0.0
MIN_VALIDATION_USES_TO_DEMOTE_RECOVERY_SKILL = 2
ACTIONABLE_SUCCESS_MIN_EVIDENCE_SUCCESSES = 2
ACTIONABLE_SUCCESS_MIN_EVIDENCE_RATE = 0.50
VALIDATION_PROBE_SKILL_EXTRA_CLUSTERS = 8


def _exp_sort_key(exp_id: str) -> Tuple[int, str]:
    if isinstance(exp_id, str) and exp_id.startswith("E") and exp_id[1:].isdigit():
        return int(exp_id[1:]), exp_id
    return 10**9, str(exp_id)


def _record_signature(record: Dict[str, Any]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    metadata_tags = []
    for tag in record.get("task_tags", []) or []:
        tag = str(tag).strip().lower()
        if tag.startswith("data_source_") or tag.startswith("task_"):
            metadata_tags.append(tag)
    max_task_tags = MAX_TASK_TAGS
    ordered_task_tags: List[str] = []
    for tag in metadata_tags + sorted(cluster_task_tags(record)):
        if tag not in ordered_task_tags:
            ordered_task_tags.append(tag)
        if len(ordered_task_tags) >= max_task_tags:
            break
    task_tags = tuple(ordered_task_tags) or ("general",)
    tool_tags = tuple(sorted(record.get("tool_tags", []) or ["reasoning"]))
    tool_tags = tool_tags[:MAX_TOOL_TAGS]
    return task_tags, tool_tags


def _cluster_tags_compatible(record_a: Dict[str, Any], record_b: Dict[str, Any]) -> bool:
    tags_a = cluster_task_tags(record_a)
    tags_b = cluster_task_tags(record_b)
    if not tags_a or not tags_b:
        return False
    return bool(tags_a & tags_b)


def _try_content_embeddings(
    records: List[Dict[str, Any]],
    *,
    record_ids: Optional[List[str]] = None,
    embedding_experiences: Optional[ExperienceLibrary] = None,
    cache_dir: Optional[str] = None,
    experience_library_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    api_key = os.environ.get("EXPERIENCE_EMBEDDING_API_KEY")
    endpoint = os.environ.get("EXPERIENCE_EMBEDDING_ENDPOINT")
    if not api_key or not endpoint:
        return None
    try:
        from experience.retriever import ExperienceRetriever

        if embedding_experiences is not None and record_ids is not None:
            indexed_records = normalize_experience_library(embedding_experiences)
            target_ids = record_ids
        else:
            indexed_records = {
                f"cluster_{idx}": record
                for idx, record in enumerate(records)
            }
            target_ids = list(indexed_records)
        retriever = ExperienceRetriever(
            experiences=indexed_records,
            embedding_model=os.environ.get("EXPERIENCE_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_api_key=api_key,
            embedding_endpoint=endpoint,
            cache_dir=cache_dir,
            enable_cache=True,
            experience_library_path=experience_library_path,
        )
        embeddings = [
            retriever._experience_embeddings.get(exp_id)
            for exp_id in target_ids
        ]
        valid = [emb for emb in embeddings if emb is not None]
        if len(valid) == len(records):
            return np.vstack(valid)
    except Exception as e:
        print(f"  [Skill Cluster] Embedding clustering unavailable, falling back to tag clustering: {e}")
    return None


def _majority_tags(records: List[Dict[str, Any]], key: str) -> List[str]:
    from collections import Counter

    counter: Counter = Counter()
    for record in records:
        for tag in record.get(key, []):
            counter[tag] += 1
    if not counter:
        return ["general"] if key == "task_tags" else ["reasoning"]
    threshold = len(records) * 0.3
    selected = {tag for tag, cnt in counter.items() if cnt >= threshold}
    if key == "task_tags":
        selected.update(
            tag for tag, cnt in counter.items()
            if str(tag).startswith(("data_source_", "task_")) and cnt >= threshold
        )
        ordered = [tag for tag, _ in counter.most_common() if tag in selected]
        return ordered[:MAX_TASK_TAGS] or [counter.most_common(1)[0][0]]
    ordered = [tag for tag, _ in counter.most_common() if tag in selected]
    return ordered[:MAX_TOOL_TAGS] or [counter.most_common(1)[0][0]]


def _cluster_by_tags_and_content(
    experiences: ExperienceLibrary,
    similarity_threshold: Optional[float] = None,
    *,
    embedding_experiences: Optional[ExperienceLibrary] = None,
    cache_dir: Optional[str] = None,
    experience_library_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Cluster experiences by embedding similarity, with tag-only fallback."""
    similarity_threshold = (
        _settings().cluster_similarity_threshold
        if similarity_threshold is None
        else similarity_threshold
    )
    normalized = normalize_experience_library(experiences)
    items = sorted(normalized.items(), key=lambda x: _exp_sort_key(x[0]))
    if not items:
        return []

    records = [record for _, record in items]
    embeddings = _try_content_embeddings(
        records,
        record_ids=[exp_id for exp_id, _ in items],
        embedding_experiences=embedding_experiences,
        cache_dir=cache_dir,
        experience_library_path=experience_library_path,
    )

    if embeddings is None:
        groups: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], List[Tuple[str, Dict[str, Any]]]] = {}
        for exp_id, record in items:
            groups.setdefault(_record_signature(record), []).append((exp_id, record))
        return [{"signature": sig, "items": group_items} for sig, group_items in groups.items()]

    similarities = cosine_similarity_matrix(embeddings)
    assigned = [False] * len(items)
    clusters = []
    for i in range(len(items)):
        if assigned[i]:
            continue
        cluster_items = [items[i]]
        assigned[i] = True
        for j in range(i + 1, len(items)):
            if (
                not assigned[j]
                and similarities[i][j] >= similarity_threshold
                and _cluster_tags_compatible(items[i][1], items[j][1])
            ):
                cluster_items.append(items[j])
                assigned[j] = True
        cluster_records = [record for _, record in cluster_items]
        clusters.append({
            "signature": (
                tuple(_majority_tags(cluster_records, "task_tags")),
                tuple(_majority_tags(cluster_records, "tool_tags")),
            ),
            "items": cluster_items,
        })
    return clusters


def _cluster_skill_id(cluster: Dict[str, Any]) -> str:
    task_tags, tool_tags = cluster["signature"]
    exp_ids = sorted([exp_id for exp_id, _ in cluster["items"]], key=_exp_sort_key)
    representative = exp_ids[0] if exp_ids else "empty"
    digest_input = "|".join(list(task_tags) + list(tool_tags) + [representative])
    digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:8]
    return f"skill_{slugify(task_tags)}_{slugify(tool_tags)}_{digest}"

def _read_rollout_evidence(records: List[Dict[str, Any]], max_rollouts: int = 4, max_chars: int = 12000) -> str:
    chunks = []
    seen_paths = set()
    for record in records:
        for rollout in record.get("evidence_rollouts", []):
            path = rollout.get("traj_path")
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            if len(chunks) >= max_rollouts:
                break
            text = ""
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                text = "(trajectory file unavailable)"
            if len(text) > max_chars // max(1, max_rollouts):
                text = text[: max_chars // max(1, max_rollouts)] + "\n...[truncated]"
            chunks.append(
                "### {rollout_id} | success={success} | score={score}\n{content}".format(
                    rollout_id=rollout.get("rollout_id", "unknown_rollout"),
                    success=rollout.get("success"),
                    score=rollout.get("accuracy_score"),
                    content=text,
                )
            )
    return "\n\n".join(chunks) or "(No rollout evidence available.)"


def _title_from_tags(task_tags: List[str], tool_tags: List[str], skill_type: str) -> str:
    tokens = []
    for tag in task_tags:
        token = skill_name_token(tag)
        if token and token not in tokens:
            tokens.append(token)
        if len(tokens) >= 3:
            break
    if not tokens:
        for tag in tool_tags:
            token = skill_name_token(tag)
            if token and token not in tokens:
                tokens.append(token)
            if len(tokens) >= 2:
                break
    if not tokens:
        tokens = ["visual"]
    if skill_type == "recovery_action_skill" and "recovery" not in tokens:
        tokens.append("recovery")
    elif skill_type == "failure_guardrail" and "guardrail" not in tokens:
        tokens.append("guardrail")
    return " ".join(tokens).replace("_", " ").title()


def _extract_skill_title(skill_content: str) -> str:
    generic_titles = {"failure guardrails", "failure guardrail", "guardrails", "skill"}
    for line in (skill_content or "").splitlines():
        match = re.match(r"^\s*#\s+(.+?)\s*$", line)
        if match:
            title = match.group(1).strip().strip("#").strip()
            lowered = title.lower()
            if title and not lowered.startswith("skill_") and lowered not in generic_titles:
                return title
    return ""


def _skill_id_from_title(title: str, skill_type: str, active_ids: set) -> str:
    base = slugify([title], fallback="visual_skill")
    if not base.startswith("skill_"):
        base = f"skill_{base}"
    if skill_type == "recovery_action_skill" and "recovery" not in base:
        base = f"{base}_recovery"
    if skill_type == "failure_guardrail" and "guardrail" not in base:
        base = f"{base}_guardrail"
    skill_id = base
    suffix = 2
    while skill_id in active_ids:
        skill_id = f"{base}_{suffix}"
        suffix += 1
    return skill_id


def _best_previous_skill_by_overlap(
    index: SkillIndex,
    linked_experience_ids: List[str],
    skill_type: str,
    active_ids: set,
    *,
    denominator: str = "union",
    require_validation: bool = False,
) -> Tuple[str, Dict[str, Any], float]:
    current = set(linked_experience_ids or [])
    if not current:
        return "", {}, 0.0
    best_id = ""
    best_record: Dict[str, Any] = {}
    best_score = 0.0
    for prev_id, prev in (index.get("skills", {}) or {}).items():
        if prev_id in active_ids or not isinstance(prev, dict):
            continue
        if prev.get("skill_type") != skill_type:
            continue
        stats = prev.get("stats") if isinstance(prev.get("stats"), dict) else {}
        if require_validation and int(stats.get("validation_used_count", 0) or 0) <= 0:
            continue
        previous = set(prev.get("linked_experience_ids", []) or [])
        if not previous:
            continue
        overlap = len(current & previous)
        if overlap <= 0:
            continue
        if denominator == "min":
            score = overlap / max(1, min(len(current), len(previous)))
        else:
            score = overlap / max(1, len(current | previous))
        if overlap >= 1 and score > best_score:
            best_id = prev_id
            best_record = prev
            best_score = score
    return best_id, best_record, best_score


def _find_previous_skill_by_overlap(
    index: SkillIndex,
    linked_experience_ids: List[str],
    skill_type: str,
    active_ids: set,
) -> str:
    """Keep skill identity stable when a regenerated cluster has the same support."""
    best_id, _, best_score = _best_previous_skill_by_overlap(
        index, linked_experience_ids, skill_type, active_ids
    )
    min_overlap = SKILL_ID_REUSE_MIN_LINK_OVERLAP
    return best_id if best_score >= min_overlap else ""


def _validation_stats_from_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate validation feedback already attached to linked experiences.

    Skill IDs can change when clusters are regenerated. Experience IDs are more
    stable, so linked experience validation gives the new skill a conservative
    prior instead of treating it as untested.
    """
    used = 0
    score_sum = 0.0
    pass_sum = 0.0
    last_stage = ""
    for record in records:
        stats = record.get("stats") if isinstance(record.get("stats"), dict) else {}
        count = int(stats.get("validation_used_count", 0) or 0)
        if count <= 0:
            continue
        used += count
        score_sum += float(
            stats.get(
                "validation_score_sum",
                float(stats.get("validation_average_score", 0.0) or 0.0) * count,
            )
            or 0.0
        )
        pass_sum += float(
            stats.get(
                "validation_pass_sum",
                float(stats.get("validation_pass_rate", 0.0) or 0.0) * count,
            )
            or 0.0
        )
        last_stage = str(stats.get("last_validation_stage") or last_stage)
    if used <= 0:
        return {}
    return {
        "validation_used_count": used,
        "validation_score_sum": round(score_sum, 6),
        "validation_pass_sum": round(pass_sum, 6),
        "validation_average_score": round(score_sum / used, 6),
        "validation_pass_rate": round(pass_sum / used, 6),
        "last_validation_stage": last_stage,
        "inherited_from_linked_experiences": True,
    }


def _previous_validation_stats_by_overlap(
    index: SkillIndex,
    linked_experience_ids: List[str],
    skill_type: str,
    active_ids: set,
) -> Dict[str, Any]:
    """Find validation stats from the nearest previous skill, even if the ID changed."""
    _, previous, best_score = _best_previous_skill_by_overlap(
        index,
        linked_experience_ids,
        skill_type,
        active_ids,
        denominator="min",
        require_validation=True,
    )
    min_score = SKILL_FEEDBACK_REUSE_MIN_LINK_OVERLAP
    if best_score < min_score:
        return {}
    return {
        key: (dict(previous[key]) if key == "stats" and isinstance(previous.get(key), dict) else previous.get(key))
        for key in (
            "stats",
            "validation_quality",
            "disabled",
            "disabled_reason",
            "recovery_status",
            "requires_validation",
        )
        if key in previous
    }


def _validation_should_disable_skill(used: int, quality: float, pass_rate: float) -> bool:
    min_disable_uses = VALIDATION_MIN_USES_TO_DISABLE
    disable_avg = VALIDATION_DISABLE_AVERAGE_THRESHOLD
    disable_pass = VALIDATION_DISABLE_PASS_THRESHOLD
    weak_min_uses = min_disable_uses
    weak_avg = LOW_QUALITY_SKILL_MIN_VALIDATION_AVG
    weak_pass = LOW_QUALITY_SKILL_MAX_PASS_RATE
    weak_validation = used >= weak_min_uses and quality < weak_avg and pass_rate <= weak_pass
    return (used >= min_disable_uses and quality <= disable_avg and pass_rate <= disable_pass) or weak_validation


def _recovery_validation_status(used: int, quality: float, pass_rate: float) -> str:
    promote_avg = RECOVERY_PROMOTE_AVERAGE_THRESHOLD
    promote_pass = RECOVERY_PROMOTE_PASS_THRESHOLD
    reject_uses = RECOVERY_REJECT_MIN_VALIDATION_USES
    reject_avg = RECOVERY_REJECT_AVERAGE_THRESHOLD
    reject_pass = RECOVERY_REJECT_PASS_THRESHOLD
    weak_reject_uses = MIN_VALIDATION_USES_TO_DEMOTE_RECOVERY_SKILL
    weak_reject_avg = LOW_QUALITY_SKILL_MIN_VALIDATION_AVG
    weak_reject_pass = LOW_QUALITY_SKILL_MAX_PASS_RATE
    weak_recovery_validation = (
        used >= weak_reject_uses
        and quality < weak_reject_avg
        and pass_rate <= weak_reject_pass
    )
    if quality >= promote_avg and pass_rate >= promote_pass:
        return "validated"
    if (used >= reject_uses and quality <= reject_avg and pass_rate <= reject_pass) or weak_recovery_validation:
        return "rejected"
    return "candidate_unverified"


def _apply_validation_status_to_metadata(metadata: Dict[str, Any]) -> None:
    """Turn inherited validation stats into selection and tool-plan gates."""
    stats = metadata.get("stats") if isinstance(metadata.get("stats"), dict) else {}
    used = int(stats.get("validation_used_count", 0) or 0)
    if used <= 0:
        return
    quality = float(stats.get("validation_average_score", 0.0) or 0.0)
    pass_rate = float(stats.get("validation_pass_rate", 0.0) or 0.0)

    metadata["validation_quality"] = round(quality, 6)

    if _validation_should_disable_skill(used, quality, pass_rate):
        metadata["disabled"] = True
        metadata["disabled_reason"] = (
            f"validation_average_score={quality:.4f}, "
            f"validation_pass_rate={pass_rate:.4f}"
        )

    if metadata.get("skill_type") != "recovery_action_skill":
        return

    recovery_status = _recovery_validation_status(used, quality, pass_rate)
    if recovery_status == "validated":
        metadata["recovery_status"] = "validated"
        metadata["requires_validation"] = False
        metadata.pop("disabled", None)
        metadata.pop("disabled_reason", None)
    elif recovery_status == "rejected":
        metadata["recovery_status"] = "rejected"
        metadata["requires_validation"] = False
        metadata["disabled"] = True
        metadata["disabled_reason"] = (
            f"recovery_validation_average={quality:.4f}, "
            f"recovery_validation_pass_rate={pass_rate:.4f}"
        )
    else:
        metadata["recovery_status"] = "candidate_unverified"
        metadata["requires_validation"] = True
        metadata.pop("disabled", None)
        metadata.pop("disabled_reason", None)


def _has_unvalidated_supervised_correction(record: Dict[str, Any]) -> bool:
    """True when a linked correction has not yet received validation feedback."""
    if not isinstance(record, dict):
        return False
    stats = record.get("stats") if isinstance(record.get("stats"), dict) else {}
    if not stats.get("supervised_correction"):
        return False
    return int(stats.get("validation_used_count", 0) or 0) <= 0


def _reset_revised_recovery_validation(
    metadata: Dict[str, Any],
    previous: Dict[str, Any],
    updated: Dict[str, Dict[str, Any]],
) -> None:
    """Treat a recovery skill with new unvalidated corrections as a new policy.

    Validation scores belong to a concrete executable policy. When reflection
    adds a new supervised correction to a previously weak recovery skill, reusing
    the old score can prevent the revised policy from being tested.
    """
    if metadata.get("skill_type") != "recovery_action_skill" or not isinstance(previous, dict):
        return

    previous_ids = {str(exp_id) for exp_id in previous.get("linked_experience_ids", []) or []}
    current_ids = {str(exp_id) for exp_id in metadata.get("linked_experience_ids", []) or []}
    added_ids = sorted(current_ids - previous_ids)
    if not added_ids:
        return
    if not any(_has_unvalidated_supervised_correction(updated.get(exp_id, {})) for exp_id in added_ids):
        return

    old_stats = metadata.get("stats") if isinstance(metadata.get("stats"), dict) else {}
    if int(old_stats.get("validation_used_count", 0) or 0) > 0:
        metadata["previous_validation_stats"] = dict(old_stats)
    metadata["pending_validation_revision"] = True
    metadata["revision_added_experience_ids"] = added_ids
    metadata["recovery_status"] = "candidate_unverified"
    metadata["requires_validation"] = True
    metadata.pop("stats", None)
    metadata.pop("validation_quality", None)
    metadata.pop("disabled", None)
    metadata.pop("disabled_reason", None)


def _inherit_validation_feedback(
    metadata: Dict[str, Any],
    index: SkillIndex,
    linked_records: List[Dict[str, Any]],
    active_ids: set,
) -> None:
    """Preserve validation signal across skill rebuilds."""
    stats = metadata.get("stats") if isinstance(metadata.get("stats"), dict) else {}
    if int(stats.get("validation_used_count", 0) or 0) <= 0:
        previous = _previous_validation_stats_by_overlap(
            index,
            metadata.get("linked_experience_ids", []) or [],
            metadata.get("skill_type", ""),
            active_ids,
        )
        if previous:
            for key, value in previous.items():
                if value not in (None, "", [], {}):
                    metadata[key] = value

    stats = metadata.get("stats") if isinstance(metadata.get("stats"), dict) else {}
    if int(stats.get("validation_used_count", 0) or 0) <= 0:
        inherited = _validation_stats_from_records(linked_records)
        if inherited:
            metadata["stats"] = inherited

    _apply_validation_status_to_metadata(metadata)


def _rewrite_skill_frontmatter_name(skill_content: str, display_name: str) -> str:
    if not skill_content:
        return skill_content
    if re.search(r"(?m)^name:\s*.*$", skill_content):
        return re.sub(r"(?m)^name:\s*.*$", f"name: {display_name}", skill_content, count=1)
    return skill_content


def _records_tool_plan_lines(records: List[Dict[str, Any]], max_steps: int = 5) -> str:
    """Build markdown tool-plan lines from structured experiences."""
    lines: List[str] = []
    for record in records:
        for step in record.get("tool_sequence", []) or []:
            if not isinstance(step, dict):
                continue
            tool = str(step.get("tool") or "tool").strip()
            purpose = str(step.get("purpose") or "run the tool-backed check").strip()
            expected = str(step.get("expected_output") or "").strip()
            line = f"{len(lines) + 1}. `{tool}`: {purpose}"
            if expected:
                line += f" -> expect {expected}"
            if line not in lines:
                lines.append(line)
            if len(lines) >= max_steps:
                return "\n".join(lines)
    return "1. Use the tagged tool only after deriving its input from the current image and question."


def _records_bullet_lines(records: List[Dict[str, Any]], field: str, max_items: int, default: str) -> str:
    lines: List[str] = []
    for record in records:
        for item in record.get(field, []) or []:
            line = f"- {item}"
            if line not in lines:
                lines.append(line)
            if len(lines) >= max_items:
                return "\n".join(lines)
    return default


def _fallback_skill_content(skill_id: str, records: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
    linked = metadata.get("linked_experience_ids", [])
    task_tags = metadata.get("task_tags", [])
    tool_tags = metadata.get("tool_tags", [])
    skill_type = metadata.get("skill_type", "action_skill")
    description = (
        "Failure-recovery action skill"
        if skill_type == "recovery_action_skill"
        else "Auto-generated fallback skill"
    )
    experience_lines = "\n".join(
        f"- {format_experience_for_prompt(record.get('id', f'E{i}'), record)}"
        for i, record in enumerate(records)
    )
    tool_plan = _records_tool_plan_lines(records)
    validation = _records_bullet_lines(
        records,
        "validation_checks",
        6,
        DEFAULT_VALIDATION_LINE,
    )
    fallback = _records_bullet_lines(
        records,
        "fallback_plan",
        4,
        DEFAULT_FALLBACK_LINE,
    )
    return FALLBACK_SKILL_MARKDOWN_TEMPLATE.format(
        skill_id=skill_id,
        description=description,
        task_label=", ".join(task_tags) or "general",
        task_tags=task_tags,
        tool_tags=tool_tags,
        linked=linked,
        tool_plan=tool_plan,
        validation=validation,
        fallback=fallback,
        experience_lines=experience_lines,
    )


def _failure_guardrail_skill_content(skill_id: str, records: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
    """For all-failure clusters, create guardrails instead of inventing a workflow."""
    linked = metadata.get("linked_experience_ids", [])
    task_tags = metadata.get("task_tags", [])
    tool_tags = metadata.get("tool_tags", [])
    experience_lines = []
    for i, record in enumerate(records):
        exp_id = record.get("id") or (linked[i] if i < len(linked) else f"E{i}")
        experience_lines.append(f"- [{exp_id}] {record.get('content', '')}")
    validation = _records_bullet_lines(
        records,
        "validation_checks",
        6,
        DEFAULT_VALIDATION_LINE,
    )
    fallback = _records_bullet_lines(
        records,
        "fallback_plan",
        4,
        DEFAULT_FALLBACK_LINE,
    )
    return FAILURE_GUARDRAIL_SKILL_MARKDOWN_TEMPLATE.format(
        skill_id=skill_id,
        task_label=", ".join(task_tags) or "similar",
        task_tags=task_tags,
        tool_tags=tool_tags,
        linked=linked,
        guardrails=os.linesep.join(experience_lines),
        validation=validation,
        fallback=fallback,
    )


def _cap_experience_count(experiences: ExperienceLibrary) -> ExperienceLibrary:
    """Keep recovery synthesis from growing the library past the configured cap."""
    limit = int(os.environ.get("EXPERIENCE_MAX_ITEMS", "0") or 0)
    if limit <= 0 or len(experiences) <= limit:
        return experiences

    def _numeric_id(exp_id: str) -> int:
        text = str(exp_id)
        return int(text[1:]) if text.startswith("E") and text[1:].isdigit() else 0

    def _prune_rank_key(item: tuple[str, Dict[str, Any]]) -> tuple[int, int, int]:
        exp_id, record = item
        stats = record.get("stats") if isinstance(record.get("stats"), dict) else {}
        validation_used = int(stats.get("validation_used_count", 0) or 0)
        status = record.get("recovery_status") or stats.get("recovery_status")
        if status == "validated" or validation_used > 0:
            bucket = 3
        elif status == "candidate_unverified":
            bucket = 0
        elif record.get("kind") == "failure":
            bucket = 1
        else:
            bucket = 2
        return (bucket, validation_used, -_numeric_id(exp_id))

    kept = dict(experiences)
    for exp_id, _ in sorted(kept.items(), key=_prune_rank_key):
        if len(kept) <= limit:
            break
        del kept[exp_id]
    if len(kept) < len(experiences):
        print(f"  [Skill Cluster] Pruned experience library to cap: {len(kept)} (<= {limit})")
    return normalize_experience_library(kept)


def _mixed_task_record(record: Dict[str, Any]) -> bool:
    task_values = {
        str(tag).strip().lower()[len("task_"):]
        for tag in record.get("task_tags", []) or []
        if str(tag).strip().lower().startswith("task_")
    }
    return len(task_values) > 1


def _rollout_success_evidence(record: Dict[str, Any]) -> tuple[int, int]:
    evidence = record.get("evidence_rollouts")
    if not isinstance(evidence, list) or not evidence:
        return 0, 0
    success = 0
    total = 0
    for item in evidence:
        if not isinstance(item, dict):
            continue
        total += 1
        score = float(item.get("accuracy_score", 0.0) or 0.0)
        if item.get("success") is True or score > 0.0:
            success += 1
    return success, total


def _is_actionable_success_record(record: Dict[str, Any]) -> bool:
    if record.get("kind") != "success":
        return False
    if experience_failed_validation(record):
        return False

    stats = record.get("stats") if isinstance(record.get("stats"), dict) else {}
    is_unverified_recovery = (
        stats.get("recovered_from_failure")
        and (record.get("recovery_status") or stats.get("recovery_status")) == "candidate_unverified"
    )
    if is_unverified_recovery:
        has_tool_plan = bool(record.get("tool_sequence") or record.get("code_template"))
        has_tool = any(
            str(tool).strip() and str(tool).strip() != "reasoning"
            for tool in record.get("tool_tags", []) or []
        )
        return has_tool_plan and has_tool

    evidence_success, evidence_total = _rollout_success_evidence(record)
    if evidence_total:
        min_success = ACTIONABLE_SUCCESS_MIN_EVIDENCE_SUCCESSES
        min_rate = ACTIONABLE_SUCCESS_MIN_EVIDENCE_RATE
        return evidence_success >= min_success and (evidence_success / max(1, evidence_total)) >= min_rate
    return True


def _cluster_support_count(cluster: Dict[str, Any]) -> int:
    total = 0
    for _, record in cluster.get("items", []) or []:
        stats = record.get("stats") if isinstance(record.get("stats"), dict) else {}
        total += int(stats.get("support_count", 1) or 1)
    return total


def _cluster_validation_signal(cluster: Dict[str, Any]) -> tuple[int, float, float]:
    """Aggregate validation feedback for cluster ranking."""
    used = 0
    score_sum = 0.0
    pass_sum = 0.0
    for _, record in cluster.get("items", []) or []:
        stats = record.get("stats") if isinstance(record.get("stats"), dict) else {}
        count = int(stats.get("validation_used_count", 0) or 0)
        if count <= 0:
            continue
        used += count
        score_sum += float(stats.get("validation_average_score", 0.0) or 0.0) * count
        pass_sum += float(stats.get("validation_pass_rate", 0.0) or 0.0) * count
    if used <= 0:
        return 0, 0.0, 0.0
    return used, score_sum / used, pass_sum / used


def _cluster_rank_key(cluster: Dict[str, Any]) -> Tuple[Any, ...]:
    validation_used, validation_score, validation_pass = _cluster_validation_signal(cluster)
    success_count = sum(1 for _, record in cluster.get("items", []) or [] if record.get("kind") == "success")
    has_positive_validation = validation_used > 0 and (validation_score > 0.0 or validation_pass > 0.0)
    return (
        success_count > 0,
        has_positive_validation,
        min(validation_used, 3),
        success_count,
        _cluster_support_count(cluster),
        len(cluster["items"]),
    )


def _cluster_validation_probe_count(cluster: Dict[str, Any]) -> int:
    """Count singleton recovery hypotheses produced from validation failures."""
    count = 0
    for _, record in cluster.get("items", []) or []:
        stats = record.get("stats") if isinstance(record.get("stats"), dict) else {}
        is_recovery = bool(stats.get("recovered_from_failure")) or bool(record.get("recovery_status"))
        is_reflection = bool(stats.get("supervised_correction")) or "validation_reflection" in {
            str(tag).strip().lower() for tag in record.get("task_tags", []) or []
        }
        if is_recovery and is_reflection and _is_actionable_success_record(record):
            count += 1
    return count


def _is_extra_validation_probe_cluster(cluster: Dict[str, Any], min_exps: int) -> bool:
    """Keep validation probes from displacing mature skill clusters."""
    if _cluster_validation_probe_count(cluster) <= 0:
        return False
    if len(cluster["items"]) >= min_exps:
        return False
    if _cluster_support_count(cluster) >= min_exps:
        return False
    return True


def _select_skill_clusters(
    clusters: List[Dict[str, Any]],
    *,
    max_clusters: int,
    min_exps: int,
) -> List[Dict[str, Any]]:
    eligible = [
        cluster
        for cluster in clusters
        if (
            len(cluster["items"]) >= min_exps
            or _cluster_validation_probe_count(cluster) > 0
            or _cluster_support_count(cluster) >= min_exps
        )
    ]
    normal_clusters = [cluster for cluster in eligible if not _is_extra_validation_probe_cluster(cluster, min_exps)]
    probe_clusters = [cluster for cluster in eligible if _is_extra_validation_probe_cluster(cluster, min_exps)]
    normal_clusters.sort(key=lambda cluster: str(cluster["signature"]))
    normal_clusters.sort(key=_cluster_rank_key, reverse=True)
    probe_clusters.sort(key=lambda cluster: (-_cluster_validation_probe_count(cluster), str(cluster["signature"])))
    max_probe_clusters = VALIDATION_PROBE_SKILL_EXTRA_CLUSTERS
    return normal_clusters[:max_clusters] + probe_clusters[:max(0, max_probe_clusters)]


def _cluster_record_groups(exp_ids: List[str], updated: ExperienceLibrary) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str], List[str]]:
    record_by_id = {exp_id: updated[exp_id] for exp_id in exp_ids if exp_id in updated}
    actionable_success_ids = [
        exp_id for exp_id, record in record_by_id.items()
        if _is_actionable_success_record(record)
    ]
    recovery_candidate_ids = [
        exp_id for exp_id in actionable_success_ids
        if (record_by_id[exp_id].get("stats") or {}).get("recovered_from_failure")
    ]
    real_success_ids = [exp_id for exp_id in actionable_success_ids if exp_id not in recovery_candidate_ids]
    return record_by_id, actionable_success_ids, recovery_candidate_ids, real_success_ids


def _cluster_source_ids(
    *,
    exp_ids: List[str],
    record_by_id: Dict[str, Dict[str, Any]],
    actionable_success_ids: List[str],
    recovery_candidate_ids: List[str],
    real_success_ids: List[str],
    updated: ExperienceLibrary,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    source_failure_ids = sorted({
        str(source_id)
        for exp_id in recovery_candidate_ids
        for source_id in record_by_id[exp_id].get("source_failure_ids", []) or []
        if str(source_id) in updated
    } | {
        exp_id for exp_id, record in record_by_id.items() if record.get("kind") == "failure"
    })
    source_success_ids = sorted({
        str(source_id)
        for exp_id in recovery_candidate_ids
        for source_id in record_by_id[exp_id].get("source_success_ids", []) or []
        if str(source_id) in updated
    } | set(real_success_ids))

    skill_source_ids = actionable_success_ids if actionable_success_ids else exp_ids
    if recovery_candidate_ids:
        skill_source_ids = list(dict.fromkeys(recovery_candidate_ids + source_success_ids + source_failure_ids))
    guardrail_source_ids = [
        exp_id for exp_id in exp_ids
        if record_by_id.get(exp_id, {}).get("kind") == "failure"
    ] or exp_ids
    return skill_source_ids, guardrail_source_ids, source_failure_ids, source_success_ids


def _cluster_skill_type(records: List[Dict[str, Any]], actionable_success_ids: List[str], recovery_candidate_ids: List[str]) -> str:
    if all(record.get("kind") == "failure" for record in records) or not actionable_success_ids:
        return "failure_guardrail"
    if recovery_candidate_ids:
        return "recovery_action_skill"
    return "action_skill"


def _metadata_support_counts(
    records: List[Dict[str, Any]],
    record_by_id: Dict[str, Dict[str, Any]],
    recovery_candidate_ids: List[str],
    source_success_ids: List[str],
) -> Dict[str, int]:
    return {
        "success": sum(record.get("kind") == "success" for record in records),
        "failure": sum(record.get("kind") == "failure" for record in records),
        "recovery": len(recovery_candidate_ids),
        "success_prior": len(source_success_ids),
        "supervised_correction": sum(
            1
            for exp_id in recovery_candidate_ids
            if (record_by_id[exp_id].get("stats") or {}).get("supervised_correction")
        ),
    }


def _cluster_skill_metadata(
    cluster: Dict[str, Any],
    updated: ExperienceLibrary,
    out_dir: Path,
) -> Tuple[str, List[str], List[Dict[str, Any]], Dict[str, Any], List[str]]:
    provisional_skill_id = _cluster_skill_id(cluster)
    items = cluster["items"]
    exp_ids = [exp_id for exp_id, _ in items]
    records = [updated[exp_id] for exp_id in exp_ids if exp_id in updated]
    record_by_id, actionable_success_ids, recovery_candidate_ids, real_success_ids = _cluster_record_groups(exp_ids, updated)
    skill_source_ids, guardrail_source_ids, source_failure_ids, source_success_ids = _cluster_source_ids(
        exp_ids=exp_ids,
        record_by_id=record_by_id,
        actionable_success_ids=actionable_success_ids,
        recovery_candidate_ids=recovery_candidate_ids,
        real_success_ids=real_success_ids,
        updated=updated,
    )
    signature_task_tags, signature_tool_tags = cluster.get("signature", ((), ()))
    task_tags = list(signature_task_tags) or sorted({tag for record in records for tag in record.get("task_tags", [])})
    tool_tags = list(signature_tool_tags) or sorted({tag for record in records for tag in record.get("tool_tags", [])})
    skill_type = _cluster_skill_type(records, actionable_success_ids, recovery_candidate_ids)

    fallback_title = _title_from_tags(task_tags, tool_tags, skill_type)
    linked_experience_ids = (
        skill_source_ids
        if skill_type in {"action_skill", "recovery_action_skill"}
        else guardrail_source_ids
    )
    metadata = {
        "skill_id": provisional_skill_id,
        "name": fallback_title,
        "description": (
            f"{skill_type.replace('_', ' ')} for "
            f"{', '.join(task_tags) or 'general'} tasks using {', '.join(tool_tags) or 'reasoning'}"
        ),
        "path": str(out_dir / f"{provisional_skill_id}.md"),
        "task_tags": task_tags,
        "tool_tags": tool_tags,
        "linked_experience_ids": linked_experience_ids,
        "support_counts": _metadata_support_counts(records, record_by_id, recovery_candidate_ids, source_success_ids),
        "skill_type": skill_type,
        "requires_tool_plan": skill_type in {"action_skill", "recovery_action_skill"},
        "recovery_status": "candidate_unverified" if skill_type == "recovery_action_skill" else "",
        "requires_validation": skill_type == "recovery_action_skill",
        "source_failure_experience_ids": source_failure_ids if skill_type == "recovery_action_skill" else [],
        "source_success_experience_ids": source_success_ids if skill_type == "recovery_action_skill" else [],
        "trigger_text": " ".join(task_tags + tool_tags),
        "updated_at": datetime.now().isoformat(),
    }
    return provisional_skill_id, exp_ids, records, metadata, skill_source_ids


def _build_skill_generation_prompt(
    *,
    provisional_skill_id: str,
    exp_ids: List[str],
    skill_source_ids: List[str],
    updated: ExperienceLibrary,
    metadata: Dict[str, Any],
) -> str:
    skill_type = metadata.get("skill_type", "action_skill")
    source_ids = skill_source_ids if skill_type in {"action_skill", "recovery_action_skill"} else exp_ids
    skill_source_records = [updated[exp_id] for exp_id in skill_source_ids if exp_id in updated]
    exp_text = "\n\n".join(
        f"[{exp_id}] {format_experience_for_prompt(exp_id, updated[exp_id])}"
        for exp_id in source_ids
        if exp_id in updated
    )
    return GENERATE_CLUSTER_SKILL_PROMPT.format(
        skill_id=provisional_skill_id,
        task_tags=", ".join(metadata.get("task_tags", [])) or "general",
        tool_tags=", ".join(metadata.get("tool_tags", [])) or "reasoning",
        experience_ids=", ".join(exp_ids),
        experiences=exp_text,
        rollout_evidence=_read_rollout_evidence(skill_source_records),
    )


def _skill_generation_fingerprint(prompt: str, llm: Any) -> str:
    payload = {
        "version": SKILL_GENERATION_CACHE_VERSION,
        "prompt": prompt,
        "model_name": str(getattr(llm, "model_name", "") or ""),
        "fallback_model_name": str(getattr(llm, "critique_model_name", "") or ""),
        "max_tokens": max(1, int(runtime_setting("skill.generation_max_tokens", 4096))),
        "fallback_max_tokens": max(1, int(runtime_setting("skill.generation_fallback_max_tokens", 4096))),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _generate_skill_content(
    *,
    provisional_skill_id: str,
    exp_ids: List[str],
    records: List[Dict[str, Any]],
    skill_source_ids: List[str],
    updated: ExperienceLibrary,
    metadata: Dict[str, Any],
    llm: Any,
    prompt: Optional[str] = None,
) -> str:
    skill_type = metadata.get("skill_type", "action_skill")
    fallback_title = metadata.get("name") or _title_from_tags(
        metadata.get("task_tags", []),
        metadata.get("tool_tags", []),
        skill_type,
    )
    prompt = prompt or _build_skill_generation_prompt(
        provisional_skill_id=provisional_skill_id,
        exp_ids=exp_ids,
        skill_source_ids=skill_source_ids,
        updated=updated,
        metadata=metadata,
    )

    if skill_type == "failure_guardrail":
        guardrail_records = [record for record in records if record.get("kind") == "failure"] or records
        return _failure_guardrail_skill_content(fallback_title, guardrail_records, metadata)
    if llm is None:
        return _fallback_skill_content(fallback_title, records, metadata)

    try:
        skill_content = llm.chat(
            prompt,
            max_tokens=max(1, int(runtime_setting("skill.generation_max_tokens", 4096))),
            max_retries=max(1, int(runtime_setting("skill.generation_reasoner_retries", 1))),
        ).strip()
    except Exception as reasoner_error:
        print(
            f"  [Skill Cluster] Reasoner failed for {provisional_skill_id}: "
            f"{reasoner_error}; using chat fallback."
        )
        try:
            skill_content = llm.chat(
                prompt,
                max_tokens=max(1, int(runtime_setting("skill.generation_fallback_max_tokens", 4096))),
                model_name=getattr(llm, "critique_model_name", None),
                timeout=max(1, int(runtime_setting("skill.generation_fallback_timeout", 90))),
                max_retries=max(1, int(runtime_setting("skill.generation_fallback_retries", 2))),
            ).strip()
        except Exception as fallback_error:
            print(
                f"  [Skill Cluster] Warning: failed to generate {provisional_skill_id} "
                f"with reasoner and chat fallback: {fallback_error}"
            )
            skill_content = ""

    try:
        skill_content = re.sub(r"^```markdown\n", "", skill_content)
        skill_content = re.sub(r"^```\n", "", skill_content)
        skill_content = re.sub(r"\n```$", "", skill_content).strip()
    except Exception as e:
        print(f"  [Skill Cluster] Warning: failed to clean {provisional_skill_id}: {e}")
        skill_content = ""
    return skill_content or _fallback_skill_content(fallback_title, records, metadata)


def _skill_generation_sources(experiences: ExperienceLibrary) -> ExperienceLibrary:
    return {
        exp_id: record
        for exp_id, record in experiences.items()
        if not experience_failed_validation(record)
        and not experience_low_validation_quality(record)
        and not _mixed_task_record(record)
    }


def _skill_source_fingerprint(
    experiences: ExperienceLibrary,
    *,
    args: Any = None,
    llm: Any = None,
) -> str:
    stable = {}
    for exp_id, record in sorted(experiences.items(), key=lambda item: _exp_sort_key(item[0])):
        item = dict(record)
        item.pop("skill_ids", None)
        stats = dict(item.get("stats") or {})
        for key in ("used_count", "success_after_used", "failure_after_used"):
            stats.pop(key, None)
        item["stats"] = stats
        stable[exp_id] = item

    model_name = str(
        getattr(llm, "model_name", "")
        or os.environ.get("EXPERIENCE_MODEL_NAME", "")
    )
    fallback_model_name = str(
        getattr(llm, "critique_model_name", "")
        or os.environ.get("EXPERIENCE_CRITIQUE_MODEL_NAME", "")
        or model_name
    )
    policy = {
        "version": SKILL_GENERATION_CACHE_VERSION,
        "prompt_template_hash": hashlib.sha256(
            GENERATE_CLUSTER_SKILL_PROMPT.encode("utf-8")
        ).hexdigest(),
        "max_clusters": int(getattr(args, "skill_cluster_max_clusters", 16) or 16),
        "min_experiences": int(getattr(args, "skill_cluster_min_experiences", 2) or 2),
        "similarity_threshold": float(_settings().cluster_similarity_threshold),
        "model_name": model_name,
        "fallback_model_name": fallback_model_name,
        "generation_max_tokens": max(1, int(runtime_setting("skill.generation_max_tokens", 4096))),
        "fallback_max_tokens": max(1, int(runtime_setting("skill.generation_fallback_max_tokens", 4096))),
        "embedding_model": os.environ.get("EXPERIENCE_EMBEDDING_MODEL", "text-embedding-3-small"),
        "embedding_enabled": bool(
            os.environ.get("EXPERIENCE_EMBEDDING_API_KEY")
            and os.environ.get("EXPERIENCE_EMBEDDING_ENDPOINT")
        ),
        "failure_recovery_enabled": bool(getattr(args, "ablation_failure_recovery", True)),
    }
    encoded = json.dumps(
        {"experiences": stable, "policy": policy},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def skill_generation_is_current(
    experiences: ExperienceLibrary,
    skill_library_path: str,
    *,
    args: Any = None,
    llm: Any = None,
) -> bool:
    normalized = _cap_experience_count(normalize_experience_library(experiences))
    source_records = _skill_generation_sources(normalized)
    index = load_skill_index(skill_library_path)
    if index.get("source_experience_fingerprint") != _skill_source_fingerprint(
        source_records,
        args=args,
        llm=llm,
    ):
        return False
    for item in (index.get("skills", {}) or {}).values():
        if not isinstance(item, dict):
            return False
        path = Path(str(item.get("path") or ""))
        if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
            return False
    return True


def _reset_experience_skill_links(experiences: ExperienceLibrary) -> ExperienceLibrary:
    updated = {}
    for exp_id, record in experiences.items():
        cleaned = dict(record)
        cleaned["skill_ids"] = []
        updated[exp_id] = normalize_experience_record(exp_id, cleaned)
    return updated


def _copy_previous_skill_metadata(metadata: Dict[str, Any], previous: Dict[str, Any]) -> None:
    if not isinstance(previous, dict):
        return
    for key in _PREVIOUS_SKILL_METADATA_KEYS:
        if key in previous:
            metadata[key] = previous[key]


def _attach_skill_to_experiences(
    updated: ExperienceLibrary,
    linked_experience_ids: List[str],
    skill_id: str,
) -> None:
    for exp_id in linked_experience_ids:
        if exp_id not in updated:
            continue
        record = dict(updated[exp_id])
        record["skill_ids"] = list(dict.fromkeys(record.get("skill_ids", []) + [skill_id]))
        updated[exp_id] = normalize_experience_record(exp_id, record)


def _write_cluster_skill(
    cluster: Dict[str, Any],
    *,
    updated: ExperienceLibrary,
    index: SkillIndex,
    active_skill_ids: set,
    out_dir: Path,
    llm: Any,
) -> str:
    prepared = _prepare_cluster_skill(cluster, updated, out_dir, llm)
    skill_content = _generate_prepared_skill(prepared, updated=updated, llm=llm)
    return _write_prepared_skill(
        prepared,
        skill_content,
        updated=updated,
        index=index,
        active_skill_ids=active_skill_ids,
        out_dir=out_dir,
    )


def _prepare_cluster_skill(
    cluster: Dict[str, Any],
    updated: ExperienceLibrary,
    out_dir: Path,
    llm: Any,
) -> Dict[str, Any]:
    provisional_skill_id, exp_ids, records, metadata, skill_source_ids = _cluster_skill_metadata(
        cluster, updated, out_dir
    )
    prompt = _build_skill_generation_prompt(
        provisional_skill_id=provisional_skill_id,
        exp_ids=exp_ids,
        skill_source_ids=skill_source_ids,
        updated=updated,
        metadata=metadata,
    )
    metadata.update({
        "generation_fingerprint": _skill_generation_fingerprint(prompt, llm),
        "generation_version": SKILL_GENERATION_CACHE_VERSION,
        "generation_model_name": str(getattr(llm, "model_name", "") or ""),
        "generation_fallback_model_name": str(getattr(llm, "critique_model_name", "") or ""),
    })
    return {
        "provisional_skill_id": provisional_skill_id,
        "exp_ids": exp_ids,
        "records": records,
        "metadata": metadata,
        "skill_source_ids": skill_source_ids,
        "prompt": prompt,
    }


def _generate_prepared_skill(
    prepared: Dict[str, Any],
    *,
    updated: ExperienceLibrary,
    llm: Any,
) -> str:
    return _generate_skill_content(
        provisional_skill_id=prepared["provisional_skill_id"],
        exp_ids=prepared["exp_ids"],
        records=prepared["records"],
        skill_source_ids=prepared["skill_source_ids"],
        updated=updated,
        metadata=prepared["metadata"],
        llm=llm,
        prompt=prepared["prompt"],
    )


def _cached_skill_for_prepared(
    prepared: Dict[str, Any],
    index: SkillIndex,
    reserved_skill_ids: set,
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    metadata = prepared["metadata"]
    linked_ids = set(metadata.get("linked_experience_ids", []) or [])
    fingerprint = metadata.get("generation_fingerprint")
    for skill_id, previous in (index.get("skills", {}) or {}).items():
        if skill_id in reserved_skill_ids or not isinstance(previous, dict):
            continue
        if previous.get("skill_type") != metadata.get("skill_type"):
            continue
        if set(previous.get("linked_experience_ids", []) or []) != linked_ids:
            continue
        previous_fingerprint = previous.get("generation_fingerprint")
        if previous_fingerprint:
            if previous_fingerprint != fingerprint:
                continue
        elif not (
            set(previous.get("task_tags", []) or []) == set(metadata.get("task_tags", []) or [])
            and set(previous.get("tool_tags", []) or []) == set(metadata.get("tool_tags", []) or [])
            and dict(previous.get("support_counts", {}) or {}) == dict(metadata.get("support_counts", {}) or {})
        ):
            continue
        path = Path(str(previous.get("path") or ""))
        if not path.exists() or not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if content:
            return skill_id, content, previous
    return None


def _assign_cached_skills(prepared_skills: List[Dict[str, Any]], index: SkillIndex) -> int:
    reserved_skill_ids = set()
    reused = 0
    for prepared in prepared_skills:
        cached = _cached_skill_for_prepared(prepared, index, reserved_skill_ids)
        if cached is None:
            continue
        skill_id, content, previous = cached
        prepared["cached_skill_id"] = skill_id
        prepared["cached_skill_content"] = content
        prepared["cached_previous"] = previous
        reserved_skill_ids.add(skill_id)
        reused += 1
    return reused


def _write_prepared_skill(
    prepared: Dict[str, Any],
    skill_content: str,
    *,
    updated: ExperienceLibrary,
    index: SkillIndex,
    active_skill_ids: set,
    out_dir: Path,
) -> str:
    metadata = prepared["metadata"]
    provisional_skill_id = prepared["provisional_skill_id"]
    skill_type = metadata.get("skill_type", "action_skill")
    cached_skill_id = str(prepared.get("cached_skill_id") or "")
    cached_previous = prepared.get("cached_previous") if isinstance(prepared.get("cached_previous"), dict) else {}
    display_name = (
        str(cached_previous.get("name") or "").strip()
        or _extract_skill_title(skill_content)
        or metadata.get("name")
        or provisional_skill_id
    )
    skill_id = cached_skill_id or _find_previous_skill_by_overlap(
        index,
        metadata.get("linked_experience_ids", []),
        skill_type,
        active_skill_ids,
    ) or _skill_id_from_title(display_name, skill_type, active_skill_ids)
    active_skill_ids.add(skill_id)

    skill_path = str(cached_previous.get("path") or out_dir / f"{skill_id}.md")
    metadata.update({
        "skill_id": skill_id,
        "name": display_name,
        "path": skill_path,
    })
    previous = index.get("skills", {}).get(skill_id, {})
    _copy_previous_skill_metadata(metadata, previous)
    if cached_skill_id and previous.get("updated_at"):
        metadata["updated_at"] = previous["updated_at"]
    _inherit_validation_feedback(
        metadata,
        index,
        [updated[exp_id] for exp_id in metadata.get("linked_experience_ids", []) or [] if exp_id in updated],
        active_skill_ids,
    )
    _reset_revised_recovery_validation(metadata, previous, updated)

    if not cached_skill_id:
        skill_path_obj = Path(metadata["path"])
        tmp_path = skill_path_obj.with_suffix(skill_path_obj.suffix + ".tmp")
        tmp_path.write_text(
            _rewrite_skill_frontmatter_name(skill_content, display_name),
            encoding="utf-8",
        )
        tmp_path.replace(skill_path_obj)
    index["skills"][skill_id] = metadata
    _attach_skill_to_experiences(updated, metadata.get("linked_experience_ids", []), skill_id)
    return skill_id


def cluster_experiences_to_skills(
    experiences: ExperienceLibrary,
    skill_library_path: str,
    llm: Any,
    args: Any = None,
) -> ExperienceLibrary:
    """Generate/update modular skills from canonical experience records."""
    normalized = normalize_experience_library(experiences)
    if not normalized or not skill_library_path:
        return normalized
    if bool(getattr(args, "ablation_failure_recovery", True)):
        normalized, _ = synthesize_recovery_experiences(normalized, llm=llm)
    normalized = _cap_experience_count(normalized)

    skill_source_records = _skill_generation_sources(normalized)
    dropped_from_skill_source = len(normalized) - len(skill_source_records)
    if dropped_from_skill_source:
        print(
            f"  [Skill Cluster] Excluded {dropped_from_skill_source} "
            "validation-failed/rejected experience(s) from skill generation."
        )

    index = load_skill_index(skill_library_path)
    index.setdefault("schema_version", 1)
    index.setdefault("skills", {})

    out_dir = skills_dir(skill_library_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_clusters = int(getattr(args, "skill_cluster_max_clusters", 16) or 16)
    min_exps = int(getattr(args, "skill_cluster_min_experiences", 2) or 2)
    clusters = _select_skill_clusters(
        _cluster_by_tags_and_content(
            skill_source_records,
            embedding_experiences=normalized,
            cache_dir=(
                os.path.dirname(args.experience_library)
                if getattr(args, "experience_library", None)
                else None
            ),
            experience_library_path=getattr(args, "experience_library", None),
        ),
        max_clusters=max_clusters,
        min_exps=min_exps,
    )

    updated = _reset_experience_skill_links(normalized)
    prepared_skills = [_prepare_cluster_skill(cluster, updated, out_dir, llm) for cluster in clusters]
    reused_skills = _assign_cached_skills(prepared_skills, index)
    if reused_skills:
        print(f"  [Skill Cluster] Reused {reused_skills}/{len(prepared_skills)} unchanged skill(s).")
    active_skill_ids = {
        str(prepared.get("cached_skill_id"))
        for prepared in prepared_skills
        if prepared.get("cached_skill_id")
    }
    pending_skills = [prepared for prepared in prepared_skills if not prepared.get("cached_skill_id")]
    generation_workers = max(
        1,
        min(len(pending_skills), int(runtime_setting("skill.generation_workers", 2))),
    ) if pending_skills else 1
    if pending_skills:
        print(f"  [Skill Cluster] Skill generation workers: {generation_workers}/{len(pending_skills)} pending")
    generated_contents: Dict[int, str] = {}
    if generation_workers > 1 and pending_skills:
        with ThreadPoolExecutor(max_workers=generation_workers) as executor:
            futures = {
                id(prepared): executor.submit(_generate_prepared_skill, prepared, updated=updated, llm=llm)
                for prepared in pending_skills
            }
            for prepared in pending_skills:
                generated_contents[id(prepared)] = futures[id(prepared)].result()
    elif pending_skills:
        for prepared in pending_skills:
            generated_contents[id(prepared)] = _generate_prepared_skill(prepared, updated=updated, llm=llm)

    for prepared in prepared_skills:
        skill_content = str(prepared.get("cached_skill_content") or generated_contents.get(id(prepared), ""))
        _write_prepared_skill(
            prepared,
            skill_content,
            updated=updated,
            index=index,
            active_skill_ids=active_skill_ids,
            out_dir=out_dir,
        )
        save_skill_index(skill_library_path, index)

    prune_stale_skills(index, active_skill_ids, out_dir)
    index["source_experience_fingerprint"] = _skill_source_fingerprint(
        skill_source_records,
        args=args,
        llm=llm,
    )
    save_skill_index(skill_library_path, index)
    print(f"  [Skill Cluster] Generated/updated {len(clusters)} modular skills in {out_dir}")
    return updated
