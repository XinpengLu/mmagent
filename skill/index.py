"""Filesystem index helpers for modular skills."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

SkillIndex = Dict[str, Any]

def skill_root(skill_library_path: str) -> Path:
    return Path(skill_library_path).expanduser().resolve().parent


def skill_index_path(skill_library_path: str) -> Path:
    return skill_root(skill_library_path) / "skill_index.json"


def skills_dir(skill_library_path: str) -> Path:
    return skill_root(skill_library_path) / "skills"


def load_skill_index(skill_library_path: Optional[str]) -> SkillIndex:
    if not skill_library_path:
        return {"schema_version": 1, "skills": {}}
    path = skill_index_path(skill_library_path)
    if not path.exists():
        return {"schema_version": 1, "skills": {}}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("skills"), dict):
        raise ValueError(f"Invalid skill index payload: {path}")
    return data


def save_skill_index(skill_library_path: str, index: SkillIndex) -> None:
    path = skill_index_path(skill_library_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compact_index = {"schema_version": index.get("schema_version", 1), "skills": {}}
    if index.get("source_experience_fingerprint"):
        compact_index["source_experience_fingerprint"] = index["source_experience_fingerprint"]
    for skill_id, item in (index.get("skills", {}) or {}).items():
        if not isinstance(item, dict):
            continue
        compact_item = {}
        for key in (
            "skill_id",
            "name",
            "description",
            "path",
            "task_tags",
            "tool_tags",
            "linked_experience_ids",
            "trigger_text",
            "updated_at",
            "support_counts",
            "skill_type",
            "requires_tool_plan",
            "requires_validation",
            "recovery_status",
            "source_failure_experience_ids",
            "source_success_experience_ids",
            "validation_quality",
            "disabled",
            "disabled_reason",
            "stats",
            "needs_refine",
            "generation_fingerprint",
            "generation_version",
            "generation_model_name",
            "generation_fallback_model_name",
        ):
            value = item.get(key)
            if value not in (None, "", [], {}):
                compact_item[key] = value
        compact_index["skills"][skill_id] = compact_item
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(compact_index, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def prune_stale_skills(index: SkillIndex, active_skill_ids: set, out_dir: Path) -> None:
    """Remove skills that no longer correspond to current experience clusters."""
    skills = index.get("skills", {})
    if not isinstance(skills, dict):
        index["skills"] = {}
        return

    out_dir_resolved = out_dir.resolve()
    for skill_id, item in list(skills.items()):
        if skill_id in active_skill_ids:
            continue
        path = item.get("path") if isinstance(item, dict) else None
        if path:
            try:
                skill_path = Path(path).resolve()
                skill_path.relative_to(out_dir_resolved)
                skill_path.unlink(missing_ok=True)
            except Exception:
                pass
        del skills[skill_id]
