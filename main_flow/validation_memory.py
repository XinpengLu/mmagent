"""Validation-driven memory selection and feedback."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

from validation_tools.history import read_history
from validation_tools.metrics import sample_metric
from utils.artifacts import MEMORY_USAGE_FILENAME
from utils.artifacts import read_json, write_json
from utils.env import env_float, env_int


_VALIDATION_STAT_KEYS = {
    "validation_used_count",
    "validation_score_sum",
    "validation_pass_sum",
    "validation_average_score",
    "validation_pass_rate",
    "validation_success_after_used",
    "validation_failure_after_used",
    "last_validation_stage",
}


def _empty_update_stats() -> Dict[str, float]:
    return {"used": 0, "pass_sum": 0.0, "score_sum": 0.0}


def _fix_skill_paths_in_memory(memory_root: Path) -> None:
    index_path = memory_root / "skill_index.json"
    skills_dir = memory_root / "skills"
    if not index_path.exists():
        return
    index = read_json(index_path)
    skills = index.get("skills", {}) if isinstance(index, dict) else {}
    for skill_id, item in skills.items():
        if not isinstance(item, dict):
            continue
        expected = skills_dir / f"{skill_id}.md"
        if expected.exists():
            item["path"] = str(expected.resolve())
            continue
        old_path = Path(str(item.get("path", "")))
        candidate = skills_dir / old_path.name
        if candidate.exists():
            item["path"] = str(candidate.resolve())
    write_json(index_path, index, indent=2)


def select_best(
    *,
    history_path: Path,
    snapshot_root: Path,
    output_memory_root: Path,
    output_json: Path,
    split: str = "val",
    metric: str = "average_at_k",
) -> None:
    rows = [row for row in read_history(history_path) if row.get("split") == split]
    if not rows:
        raise SystemExit(f"No rows for split={split}")
    best = max(rows, key=lambda row: (float(row.get(metric, 0.0)), float(row.get("pass_at_k", 0.0))))
    stage = str(best.get("stage"))
    snapshot = snapshot_root / f"stage_{stage}"
    if not snapshot.exists():
        raise FileNotFoundError(f"Missing snapshot for best stage: {snapshot}")

    out = output_memory_root
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(snapshot, out)
    _fix_skill_paths_in_memory(out)

    result = {
        "selected_stage": stage,
        "metric": metric,
        "value": best.get(metric),
        "pass_at_k": best.get("pass_at_k"),
        "average_at_k": best.get("average_at_k"),
        "source_snapshot": str(snapshot),
        "output_memory_root": str(out),
    }
    write_json(output_json, result, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _update_validation_stats(
    stats: Dict[str, Any],
    *,
    pass_k: float,
    avg_k: float,
    stage: str,
) -> Dict[str, Any]:
    stats = dict(stats or {})
    used = int(stats.get("validation_used_count", 0) or 0) + 1
    score_sum = float(stats.get("validation_score_sum", 0.0) or 0.0) + avg_k
    pass_sum = float(stats.get("validation_pass_sum", 0.0) or 0.0) + pass_k
    stats["validation_used_count"] = used
    stats["validation_score_sum"] = round(score_sum, 6)
    stats["validation_pass_sum"] = round(pass_sum, 6)
    stats["validation_average_score"] = round(score_sum / used, 6)
    stats["validation_pass_rate"] = round(pass_sum / used, 6)
    stats["validation_success_after_used"] = int(stats.get("validation_success_after_used", 0) or 0) + int(pass_k > 0)
    stats["validation_failure_after_used"] = int(stats.get("validation_failure_after_used", 0) or 0) + int(pass_k <= 0)
    stats["last_validation_stage"] = str(stage)
    return stats


def _clear_validation_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats = dict(stats or {})
    for key in _VALIDATION_STAT_KEYS:
        stats.pop(key, None)
    return stats


def _reset_validation_state(skills: Dict[str, Any], exp_lib: Dict[str, Any]) -> None:
    """Make feedback application idempotent before replaying validation logs."""
    for item in skills.values():
        if not isinstance(item, dict):
            continue
        item["stats"] = _clear_validation_stats(item.get("stats", {}))
        item.pop("validation_quality", None)
        reason = str(item.get("disabled_reason", ""))
        if reason.startswith("validation_") or reason.startswith("recovery_validation_"):
            item.pop("disabled", None)
            item.pop("disabled_reason", None)
        if item.get("skill_type") == "recovery_action_skill":
            item["recovery_status"] = "candidate_unverified"
            item["requires_validation"] = True

    for record in exp_lib.values() if isinstance(exp_lib, dict) else []:
        if not isinstance(record, dict):
            continue
        stats = record.get("stats") if isinstance(record.get("stats"), dict) else {}
        record["stats"] = _clear_validation_stats(stats)
        if record["stats"].get("recovered_from_failure"):
            record["stats"]["recovery_status"] = "candidate_unverified"
            record["recovery_status"] = "candidate_unverified"


def _parse_int(value: object) -> int | None:
    text = str(value).strip()
    digits = text[1:] if text.startswith(("+", "-")) else text
    return int(text) if digits.isdigit() else None


def _stage_number(path: Path) -> int | None:
    return _parse_int(path.name.rsplit("_", 1)[-1])


def _validation_metric_dirs(metrics_dir: Path, stage: str) -> list[tuple[str, Path]]:
    """Return completed validation dirs up to the requested stage."""
    max_stage = _parse_int(stage)
    run_root = metrics_dir.parent
    dirs: list[tuple[int, Path]] = []
    for path in run_root.glob("val_stage_*"):
        stage_num = _stage_number(path)
        if stage_num is None:
            continue
        if max_stage is not None and stage_num > max_stage:
            continue
        if any(path.glob(f"*/{MEMORY_USAGE_FILENAME}")):
            dirs.append((stage_num, path))

    if not dirs and metrics_dir.exists():
        stage_num = _stage_number(metrics_dir)
        dirs.append((stage_num or 0, metrics_dir))

    dirs.sort(key=lambda item: item[0])
    return [(str(stage_num), path) for stage_num, path in dirs]


def _action_memory_ids(usage: Dict[str, Any]) -> list[str]:
    """Attribute validation outcomes to memories that actually drove action."""
    explicit = [str(exp_id) for exp_id in usage.get("action_experience_ids", []) or [] if str(exp_id)]
    if explicit:
        return list(dict.fromkeys(explicit))
    ids: list[str] = []
    for field in ("direct_tool_plan_experience_ids", "linked_experience_ids"):
        for exp_id in usage.get(field, []) or []:
            exp_id = str(exp_id)
            if exp_id and exp_id not in ids:
                ids.append(exp_id)
    if ids:
        return ids
    return [str(exp_id) for exp_id in usage.get("retrieved_experience_ids", []) or []]


def _ensure_skill_metadata(skills: Dict[str, Any], exp_lib: Dict[str, Any]) -> Dict[str, Any]:
    for skill_id, item in list(skills.items()):
        if not isinstance(item, dict):
            continue
        linked = [exp_lib.get(exp_id) for exp_id in item.get("linked_experience_ids", []) or []]
        linked = [record for record in linked if isinstance(record, dict)]
        success_count = sum(record.get("kind") == "success" for record in linked)
        failure_count = sum(record.get("kind") == "failure" for record in linked)
        if "support_counts" not in item:
            item["support_counts"] = {"success": success_count, "failure": failure_count}
        if "skill_type" not in item:
            item["skill_type"] = "failure_guardrail" if linked and success_count == 0 else "action_skill"
        if "requires_tool_plan" not in item:
            item["requires_tool_plan"] = item.get("skill_type") != "failure_guardrail"
        if "name" not in item:
            item["name"] = skill_id
        if "description" not in item:
            item["description"] = (
                f"{item.get('skill_type', 'skill').replace('_', ' ')} for "
                f"{', '.join(item.get('task_tags', []) or []) or 'general'} tasks"
            )
        skills[skill_id] = item
    return skills


def _validation_values(stats: Dict[str, Any]) -> tuple[int, float, float]:
    return (
        int(stats.get("validation_used_count", 0) or 0),
        float(stats.get("validation_average_score", 0.0) or 0.0),
        float(stats.get("validation_pass_rate", 0.0) or 0.0),
    )


def _weak_validation(
    used: int,
    quality: float,
    pass_rate: float,
    *,
    min_uses_env: str,
    quality_env: str,
    pass_env: str,
    default_min_uses: int,
) -> bool:
    return (
        used >= env_int(min_uses_env, default_min_uses)
        and quality < env_float(quality_env, 0.12)
        and pass_rate <= env_float(pass_env, 0.25)
    )


def _recovery_validation_status(
    stats: Dict[str, Any],
    *,
    weak_min_uses_env: str,
    weak_quality_env: str,
    weak_pass_env: str,
    reject_if_already_disabled: bool = False,
) -> tuple[str, float, float] | None:
    used, quality, pass_rate = _validation_values(stats)
    if used <= 0:
        return None
    if quality >= env_float("RECOVERY_PROMOTE_AVERAGE_THRESHOLD", 0.25) and pass_rate >= env_float("RECOVERY_PROMOTE_PASS_THRESHOLD", 0.50):
        return "validated", quality, pass_rate
    rejected = (
        reject_if_already_disabled
        or (
            used >= env_int("RECOVERY_REJECT_MIN_VALIDATION_USES", 3)
            and quality <= env_float("RECOVERY_REJECT_AVERAGE_THRESHOLD", 0.01)
            and pass_rate <= env_float("RECOVERY_REJECT_PASS_THRESHOLD", 0.0)
        )
        or _weak_validation(
            used,
            quality,
            pass_rate,
            min_uses_env=weak_min_uses_env,
            quality_env=weak_quality_env,
            pass_env=weak_pass_env,
            default_min_uses=2,
        )
    )
    return ("rejected" if rejected else "candidate_unverified"), quality, pass_rate


def _update_recovery_skill_status(item: Dict[str, Any]) -> None:
    if item.get("skill_type") != "recovery_action_skill":
        return
    decision = _recovery_validation_status(
        item.get("stats") or {},
        weak_min_uses_env="MIN_VALIDATION_USES_TO_DEMOTE_RECOVERY_SKILL",
        weak_quality_env="LOW_QUALITY_SKILL_MIN_VALIDATION_AVG",
        weak_pass_env="LOW_QUALITY_SKILL_MAX_PASS_RATE",
        reject_if_already_disabled=bool(item.get("disabled")),
    )
    if decision is None:
        return
    status, quality, pass_rate = decision
    if status == "validated":
        item["recovery_status"] = "validated"
        item["requires_validation"] = False
        item.pop("disabled", None)
        item.pop("disabled_reason", None)
        item["selection_weight"] = round(max(float(item.get("selection_weight", 1.0) or 1.0), 0.65 + quality), 6)
    elif status == "rejected":
        item["recovery_status"] = "rejected"
        item["requires_validation"] = False
        item["disabled"] = True
        item["disabled_reason"] = (
            f"recovery_validation_average={quality:.4f}, "
            f"recovery_validation_pass_rate={pass_rate:.4f}"
        )
    else:
        item["recovery_status"] = "candidate_unverified"
        item["requires_validation"] = True
        item.pop("disabled", None)
        item.pop("disabled_reason", None)


def _update_recovery_experience_status(record: Dict[str, Any]) -> None:
    stats = record.get("stats") or {}
    if not stats.get("recovered_from_failure"):
        return
    decision = _recovery_validation_status(
        stats,
        weak_min_uses_env="MIN_VALIDATION_USES_TO_DEMOTE_RECOVERY_EXPERIENCE",
        weak_quality_env="LOW_QUALITY_EXPERIENCE_MIN_VALIDATION_AVG",
        weak_pass_env="LOW_QUALITY_EXPERIENCE_MAX_PASS_RATE",
    )
    if decision is None:
        return
    status, _, _ = decision
    record["stats"]["recovery_status"] = status
    record["recovery_status"] = status


def _update_skill_validation_status(
    item: Dict[str, Any],
    *,
    min_validation_uses_to_disable: int,
    disable_average_threshold: float,
    disable_pass_threshold: float,
) -> None:
    stats = item.get("stats") if isinstance(item.get("stats"), dict) else {}
    quality = float(stats.get("validation_average_score", 0.0) or 0.0)
    pass_rate = float(stats.get("validation_pass_rate", 0.0) or 0.0)
    item["validation_quality"] = round(quality, 6)
    item["selection_weight"] = round(max(0.05, 0.65 + quality), 6)

    used = int(stats.get("validation_used_count", 0) or 0)
    weak_validation = _weak_validation(
        used,
        quality,
        pass_rate,
        min_uses_env="MIN_VALIDATION_USES_TO_DEMOTE_SKILL",
        quality_env="LOW_QUALITY_SKILL_MIN_VALIDATION_AVG",
        pass_env="LOW_QUALITY_SKILL_MAX_PASS_RATE",
        default_min_uses=3,
    )
    if (
        used >= min_validation_uses_to_disable
        and quality <= disable_average_threshold
        and pass_rate <= disable_pass_threshold
    ) or weak_validation:
        item["disabled"] = True
        item["disabled_reason"] = (
            f"validation_average_score={quality:.4f}, "
            f"validation_pass_rate={pass_rate:.4f}"
        )
    elif item.get("disabled") and quality > disable_average_threshold:
        item.pop("disabled", None)
        item.pop("disabled_reason", None)
    _update_recovery_skill_status(item)


def _add_update(updates: Dict[str, Dict[str, float]], item_id: str, pass_k: float, avg_k: float) -> None:
    upd = updates[item_id]
    upd["used"] += 1
    upd["pass_sum"] += pass_k
    upd["score_sum"] += avg_k


def _apply_skill_sample_feedback(
    skills: Dict[str, Any],
    skill_updates: Dict[str, Dict[str, float]],
    usage: Dict[str, Any],
    *,
    pass_k: float,
    avg_k: float,
    stage_name: str,
    min_validation_uses_to_disable: int,
    disable_average_threshold: float,
    disable_pass_threshold: float,
) -> None:
    for skill_id in usage.get("selected_skill_ids", []) or []:
        if skill_id not in skills or not isinstance(skills[skill_id], dict):
            continue
        item = skills[skill_id]
        item["stats"] = _update_validation_stats(
            item.get("stats", {}),
            pass_k=pass_k,
            avg_k=avg_k,
            stage=stage_name,
        )
        if item.get("pending_validation_revision"):
            item.pop("pending_validation_revision", None)
            item.pop("revision_added_experience_ids", None)
        _update_skill_validation_status(
            item,
            min_validation_uses_to_disable=min_validation_uses_to_disable,
            disable_average_threshold=disable_average_threshold,
            disable_pass_threshold=disable_pass_threshold,
        )
        skills[skill_id] = item
        _add_update(skill_updates, skill_id, pass_k, avg_k)


def _apply_experience_sample_feedback(
    exp_lib: Dict[str, Any],
    exp_updates: Dict[str, Dict[str, float]],
    usage: Dict[str, Any],
    *,
    pass_k: float,
    avg_k: float,
    stage_name: str,
) -> None:
    if not isinstance(exp_lib, dict):
        return
    for exp_id in _action_memory_ids(usage):
        if exp_id not in exp_lib or not isinstance(exp_lib[exp_id], dict):
            continue
        record = exp_lib[exp_id]
        record["stats"] = _update_validation_stats(
            record.get("stats", {}),
            pass_k=pass_k,
            avg_k=avg_k,
            stage=stage_name,
        )
        _update_recovery_experience_status(record)
        exp_lib[exp_id] = record
        _add_update(exp_updates, exp_id, pass_k, avg_k)


def _summarize_updates(updates: Dict[str, Dict[str, float]], *, skills: Dict[str, Any] | None = None) -> Dict[str, Any]:
    result = {}
    for item_id, upd in sorted(updates.items()):
        row = {
            "used": int(upd["used"]),
            "pass_rate": round(upd["pass_sum"] / upd["used"], 6) if upd["used"] else 0.0,
            "average_score": round(upd["score_sum"] / upd["used"], 6) if upd["used"] else 0.0,
        }
        if skills is not None:
            row["disabled"] = bool(skills.get(item_id, {}).get("disabled")) if isinstance(skills.get(item_id), dict) else False
        result[item_id] = row
    return result


def _load_memory_for_feedback(memory_root: Path) -> tuple[Path, Path, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    index_path = memory_root / "skill_index.json"
    exp_path = memory_root / "experiences.json"
    index = read_json(index_path) if index_path.exists() else {"schema_version": 1, "skills": {}}
    skills = index.get("skills", {}) if isinstance(index, dict) else {}
    exp_raw = read_json(exp_path) if exp_path.exists() else {"schema_version": 1, "experiences": {}}
    exp_lib = exp_raw.get("experiences", exp_raw) if isinstance(exp_raw, dict) else {}
    if isinstance(exp_lib, dict):
        skills = _ensure_skill_metadata(skills, exp_lib)
        _reset_validation_state(skills, exp_lib)
    return index_path, exp_path, index, exp_raw, {"skills": skills, "experiences": exp_lib}


def _write_feedback_memory(index_path: Path, exp_path: Path, index: Dict[str, Any], exp_raw: Dict[str, Any], skills: Dict[str, Any], exp_lib: Dict[str, Any]) -> None:
    if isinstance(index, dict):
        index["skills"] = skills
        write_json(index_path, index, indent=2)
    if isinstance(exp_raw, dict):
        if "experiences" in exp_raw:
            exp_raw["experiences"] = exp_lib
            write_json(exp_path, exp_raw, indent=2)
        else:
            write_json(exp_path, exp_lib, indent=2)


def _apply_feedback_from_metric_dirs(
    *,
    metric_dirs: list[tuple[str, Path]],
    skills: Dict[str, Any],
    exp_lib: Dict[str, Any],
    skill_updates: Dict[str, Dict[str, float]],
    exp_updates: Dict[str, Dict[str, float]],
    result: Dict[str, Any],
    k: int,
    min_validation_uses_to_disable: int,
    disable_average_threshold: float,
    disable_pass_threshold: float,
) -> None:
    for stage_name, current_metrics_dir in metric_dirs:
        for usage_path in current_metrics_dir.glob(f"*/{MEMORY_USAGE_FILENAME}"):
            usage = read_json(usage_path)
            if not isinstance(usage, dict):
                raise ValueError(f"{usage_path} must contain a JSON object.")

            pass_k, avg_k = sample_metric(usage_path.parent, k)
            result["samples"] += 1
            _apply_skill_sample_feedback(
                skills,
                skill_updates,
                usage,
                pass_k=pass_k,
                avg_k=avg_k,
                stage_name=stage_name,
                min_validation_uses_to_disable=min_validation_uses_to_disable,
                disable_average_threshold=disable_average_threshold,
                disable_pass_threshold=disable_pass_threshold,
            )
            _apply_experience_sample_feedback(
                exp_lib,
                exp_updates,
                usage,
                pass_k=pass_k,
                avg_k=avg_k,
                stage_name=stage_name,
            )


def apply_validation_feedback(
    *,
    metrics_dir: Path,
    memory_root: Path,
    stage: str,
    output_json: Path,
    k: int,
    min_validation_uses_to_disable: int,
    disable_average_threshold: float,
    disable_pass_threshold: float,
) -> None:
    """Write validation effectiveness stats back to selected skills/experiences.

    The update is replay-based instead of append-only. This keeps validation
    feedback idempotent when a long benchmark is resumed and completed
    validation stages are skipped by the inference pipeline.
    """
    index_path = memory_root / "skill_index.json"
    exp_path = memory_root / "experiences.json"
    result: Dict[str, Any] = {
        "stage": str(stage),
        "samples": 0,
        "skill_updates": {},
        "experience_updates": {},
    }

    if not index_path.exists() and not exp_path.exists():
        write_json(output_json, result, indent=2)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    index_path, exp_path, index, exp_raw, memory = _load_memory_for_feedback(memory_root)
    skills = memory["skills"]
    exp_lib = memory["experiences"]

    skill_updates: Dict[str, Dict[str, float]] = defaultdict(_empty_update_stats)
    exp_updates: Dict[str, Dict[str, float]] = defaultdict(_empty_update_stats)
    metric_dirs = _validation_metric_dirs(metrics_dir, str(stage))
    result["recomputed_stages"] = [stage_name for stage_name, _ in metric_dirs]

    _apply_feedback_from_metric_dirs(
        metric_dirs=metric_dirs,
        skills=skills,
        exp_lib=exp_lib,
        skill_updates=skill_updates,
        exp_updates=exp_updates,
        result=result,
        k=k,
        min_validation_uses_to_disable=min_validation_uses_to_disable,
        disable_average_threshold=disable_average_threshold,
        disable_pass_threshold=disable_pass_threshold,
    )
    result["skill_updates"] = _summarize_updates(skill_updates, skills=skills)
    result["experience_updates"] = _summarize_updates(exp_updates)

    _write_feedback_memory(index_path, exp_path, index, exp_raw, skills, exp_lib)

    write_json(output_json, result, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def calibrate_validation_status(*, memory_root: Path, output_json: Path | None = None) -> None:
    """Recompute validation-derived status from existing memory statistics.

    This is useful after changing promotion/rejection thresholds or resuming a
    run. It does not delete, merge, or rewrite memories; it only updates
    `recovery_status`, `disabled`, and related gate fields from accumulated
    validation counters already stored in the memory bank.
    """
    index_path = memory_root / "skill_index.json"
    exp_path = memory_root / "experiences.json"
    result: Dict[str, Any] = {
        "memory_root": str(memory_root),
        "skill_count": 0,
        "experience_count": 0,
        "disabled_skill_count_before": 0,
        "disabled_skill_count_after": 0,
        "skill_status_counts": {},
        "experience_status_counts": {},
    }

    index = read_json(index_path) if index_path.exists() else {"schema_version": 1, "skills": {}}
    skills = index.get("skills", {}) if isinstance(index, dict) else {}
    exp_raw = read_json(exp_path) if exp_path.exists() else {"schema_version": 1, "experiences": {}}
    exp_lib = exp_raw.get("experiences", exp_raw) if isinstance(exp_raw, dict) else {}

    if isinstance(skills, dict):
        result["skill_count"] = len(skills)
        result["disabled_skill_count_before"] = sum(
            1 for item in skills.values() if isinstance(item, dict) and item.get("disabled")
        )
        for item in skills.values():
            if isinstance(item, dict):
                _update_recovery_skill_status(item)
        result["disabled_skill_count_after"] = sum(
            1 for item in skills.values() if isinstance(item, dict) and item.get("disabled")
        )
        for item in skills.values():
            if isinstance(item, dict):
                status = str(item.get("recovery_status") or "none")
                result["skill_status_counts"][status] = result["skill_status_counts"].get(status, 0) + 1
        if isinstance(index, dict):
            index["skills"] = skills
            write_json(index_path, index, indent=2)

    if isinstance(exp_lib, dict):
        result["experience_count"] = len(exp_lib)
        for record in exp_lib.values():
            if isinstance(record, dict):
                _update_recovery_experience_status(record)
        for record in exp_lib.values():
            if isinstance(record, dict):
                status = str(record.get("recovery_status") or "none")
                result["experience_status_counts"][status] = result["experience_status_counts"].get(status, 0) + 1
        if isinstance(exp_raw, dict) and "experiences" in exp_raw:
            exp_raw["experiences"] = exp_lib
            write_json(exp_path, exp_raw, indent=2)
        else:
            write_json(exp_path, exp_lib, indent=2)

    if output_json:
        write_json(output_json, result, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
