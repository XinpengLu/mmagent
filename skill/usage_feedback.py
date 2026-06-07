"""Post-rollout usage feedback for skills and experiences."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.artifacts import MEMORY_USAGE_FILENAME, read_json

from experience import EXPERIENCE_LIBRARY_LOCK, load_experiences, save_library
from experience.schema import normalize_experience_record
from .index import load_skill_index, save_skill_index


def _action_memory_ids(usage: Dict[str, Any]) -> List[str]:
    """Attribute rollout outcomes to memories that actually drove action."""
    explicit = [str(exp_id) for exp_id in usage.get("action_experience_ids", []) or [] if str(exp_id)]
    if explicit:
        return list(dict.fromkeys(explicit))
    ids: List[str] = []
    for field in ("direct_tool_plan_experience_ids", "linked_experience_ids"):
        for exp_id in usage.get(field, []) or []:
            exp_id = str(exp_id)
            if exp_id and exp_id not in ids:
                ids.append(exp_id)
    if ids:
        return ids
    return [str(exp_id) for exp_id in usage.get("retrieved_experience_ids", []) or []]


def update_memory_usage_after_sample(
    experience_library_path: Optional[str],
    skill_library_path: Optional[str],
    sample_dir: str,
    sample: Dict[str, Any],
    rollout_results: List[Dict[str, Any]],
) -> None:
    usage_path = Path(sample_dir) / MEMORY_USAGE_FILENAME
    if not usage_path.exists() or not experience_library_path:
        return
    usage = read_json(usage_path)
    if not isinstance(usage, dict):
        raise ValueError(f"{usage_path} must contain a JSON object.")

    with EXPERIENCE_LIBRARY_LOCK:
        experiences = load_experiences(experience_library_path)
        selected_skill_ids = usage.get("selected_skill_ids", [])
        used_exp_ids = _action_memory_ids(usage)
        similarity_scores = usage.get("similarity_scores", {})
        success = any(
            float(result.get("accuracy_score", 0.0) or 0.0) == 1.0
            for result in rollout_results
            if isinstance(result, dict)
        )

        valid_used_exp_ids = []
        for exp_id in used_exp_ids:
            if exp_id not in experiences:
                continue
            valid_used_exp_ids.append(exp_id)
            record = dict(experiences[exp_id])
            stats = dict(record.get("stats", {}))
            stats["used_count"] = int(stats.get("used_count", 0) or 0) + 1
            key = "success_after_used" if success else "failure_after_used"
            stats[key] = int(stats.get(key, 0) or 0) + 1
            record["stats"] = stats
            conf = float(record.get("confidence", 0.5) or 0.5)
            sim = float(similarity_scores.get(exp_id, 0.5))
            if success:
                conf = min(1.0, conf + 0.05 * sim)
            else:
                conf = max(0.0, conf - 0.1 * sim)
            record["confidence"] = round(conf, 4)
            experiences[exp_id] = normalize_experience_record(exp_id, record)

        save_library(experience_library_path, experiences)

        if skill_library_path and selected_skill_ids:
            index = load_skill_index(skill_library_path)
            for skill_id in selected_skill_ids:
                item = index.get("skills", {}).get(skill_id)
                if not item:
                    continue
                stats = dict(item.get("stats", {}))
                stats["used_count"] = int(stats.get("used_count", 0) or 0) + 1
                key = "success_after_used" if success else "failure_after_used"
                stats[key] = int(stats.get(key, 0) or 0) + 1
                item["stats"] = stats
                if not success:
                    item["needs_refine"] = True
                    item["linked_experience_ids"] = list(
                        dict.fromkeys(item.get("linked_experience_ids", []) + valid_used_exp_ids)
                    )
                index["skills"][skill_id] = item
            save_skill_index(skill_library_path, index)
