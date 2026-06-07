"""Metric and memory summaries for validation stages."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from utils.artifacts import (
    AVERAGE_PASS_FILENAME,
    DATASET_SUMMARY_FILENAME,
    ROLLOUT_METRICS_FILENAME,
    SAMPLE_SUMMARY_FILENAME,
    TRAJECTORY_FILENAME,
    MEMORY_USAGE_FILENAME,
    iter_jsonl,
)
from .tasks import task_from_sample_dir
from utils.artifacts import read_json


def load_metrics(metrics_dir: Path, k: int) -> Dict[str, Any]:
    metrics_at_k = metrics_dir / DATASET_SUMMARY_FILENAME
    average_pass = metrics_dir / AVERAGE_PASS_FILENAME
    if metrics_at_k.exists():
        data = read_json(metrics_at_k)
        item = (data.get("k_metrics") or {}).get(str(k), {})
        return {
            "num_samples": data.get("num_samples"),
            "num_rollouts": data.get("num_rollouts"),
            "pass_at_k": float(item.get("pass_at_k", 0.0) or 0.0),
            "average_at_k": float(item.get("average_at_k", 0.0) or 0.0),
            "metrics_file": str(metrics_at_k),
        }
    if average_pass.exists():
        data = read_json(average_pass)
        return {
            "num_samples": None,
            "num_rollouts": None,
            "pass_at_k": float(data.get(f"pass@{k}", 0.0) or 0.0),
            "average_at_k": float(data.get(f"average@{k}", 0.0) or 0.0),
            "metrics_file": str(average_pass),
        }
    raise FileNotFoundError(f"No metrics_at_k.json or average_pass_at_k.json under {metrics_dir}")


def per_task_metrics(metrics_dir: Path, k: int) -> Dict[str, Dict[str, Any]]:
    rows = []
    for sample_summary_path in metrics_dir.glob(f"*/{SAMPLE_SUMMARY_FILENAME}"):
        data = read_json(sample_summary_path)
        task = task_from_sample_dir(sample_summary_path.parent)
        rows.append(
            (
                task,
                float(data.get(f"pass@{k}", 0.0) or 0.0),
                float(data.get(f"average@{k}", 0.0) or 0.0),
            )
        )
    agg: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for task, pass_k, avg_k in rows:
        agg[task][0] += 1
        agg[task][1] += pass_k
        agg[task][2] += avg_k
    return {
        task: {
            "num_samples": int(vals[0]),
            "pass_at_k": vals[1] / vals[0] if vals[0] else 0.0,
            "average_at_k": vals[2] / vals[0] if vals[0] else 0.0,
        }
        for task, vals in sorted(agg.items())
    }


def tool_usage(metrics_dir: Path) -> Dict[str, Any]:
    rollouts = 0
    with_tool = 0
    tool_calls = Counter()
    calls_per_rollout: List[int] = []
    acc_with: List[float] = []
    acc_without: List[float] = []
    traj_paths = list(metrics_dir.glob(f"*/rollout_*/{TRAJECTORY_FILENAME}")) + [
        path for path in metrics_dir.glob(f"*/{TRAJECTORY_FILENAME}")
        if "rollout_" not in str(path.parent.name)
    ]
    for traj in traj_paths:
        rollouts += 1
        count = 0
        for item in iter_jsonl(traj):
            call = item.get("tool_call")
            if isinstance(call, dict):
                name = call.get("tool_name") or call.get("name") or "unknown"
                tool_calls[str(name)] += 1
                count += 1
        metrics_path = traj.parent / ROLLOUT_METRICS_FILENAME
        acc = 0.0
        if metrics_path.exists():
            acc = float((read_json(metrics_path).get("accuracy_score", 0.0) or 0.0))
        if count:
            with_tool += 1
            acc_with.append(acc)
        else:
            acc_without.append(acc)
        calls_per_rollout.append(count)
    return {
        "rollouts": rollouts,
        "with_tool_rollouts": with_tool,
        "with_tool_ratio": with_tool / rollouts if rollouts else 0.0,
        "tool_calls": dict(tool_calls),
        "avg_tool_calls_per_rollout": sum(calls_per_rollout) / len(calls_per_rollout) if calls_per_rollout else 0.0,
        "avg_acc_with_tool": sum(acc_with) / len(acc_with) if acc_with else None,
        "avg_acc_without_tool": sum(acc_without) / len(acc_without) if acc_without else None,
    }


def memory_stats(memory_root: Path) -> Dict[str, Any]:
    exp_path = memory_root / "experiences.json"
    skill_index_path = memory_root / "skill_index.json"
    stats: Dict[str, Any] = {
        "experience_count": 0,
        "skill_count": 0,
        "experience_kind_counts": {},
        "task_tags_top": [],
        "tool_tags_top": [],
        "with_tool_sequence": 0,
        "with_validation_checks": 0,
        "with_fallback_plan": 0,
    }
    if exp_path.exists():
        raw = read_json(exp_path)
        exps = raw.get("experiences", raw) if isinstance(raw, dict) else raw
        vals = list(exps.values()) if isinstance(exps, dict) else list(exps or [])
        stats["experience_count"] = len(vals)
        stats["experience_kind_counts"] = dict(Counter(str(e.get("kind", "unknown")) for e in vals if isinstance(e, dict)))
        stats["task_tags_top"] = Counter(
            tag for e in vals if isinstance(e, dict) for tag in (e.get("task_tags") or [])
        ).most_common(10)
        stats["tool_tags_top"] = Counter(
            tag for e in vals if isinstance(e, dict) for tag in (e.get("tool_tags") or [])
        ).most_common(10)
        stats["with_tool_sequence"] = sum(bool(e.get("tool_sequence")) for e in vals if isinstance(e, dict))
        stats["with_validation_checks"] = sum(bool(e.get("validation_checks")) for e in vals if isinstance(e, dict))
        stats["with_fallback_plan"] = sum(bool(e.get("fallback_plan")) for e in vals if isinstance(e, dict))
    if skill_index_path.exists():
        raw = read_json(skill_index_path)
        skills = raw.get("skills", {}) if isinstance(raw, dict) else {}
        stats["skill_count"] = len(skills)
        stats["skill_used_count"] = sum(
            int((item.get("stats") or {}).get("used_count", 0) or 0)
            for item in skills.values()
            if isinstance(item, dict)
        )
        validation_qualities = [
            float(item.get("validation_quality", 0.0) or 0.0)
            for item in skills.values()
            if isinstance(item, dict) and item.get("validation_quality") is not None
        ]
        stats["disabled_skill_count"] = sum(
            1 for item in skills.values() if isinstance(item, dict) and item.get("disabled")
        )
        stats["validation_skill_count"] = len(validation_qualities)
        stats["validation_quality_avg"] = (
            sum(validation_qualities) / len(validation_qualities)
            if validation_qualities else None
        )
    return stats


def _load_experiences(memory_root: Path) -> Dict[str, Dict[str, Any]]:
    path = memory_root / "experiences.json"
    if not path.exists():
        return {}
    raw = read_json(path)
    exps = raw.get("experiences", raw) if isinstance(raw, dict) else raw
    return {str(k): v for k, v in (exps or {}).items() if isinstance(v, dict)} if isinstance(exps, dict) else {}


def _load_skills(memory_root: Path) -> Dict[str, Dict[str, Any]]:
    path = memory_root / "skill_index.json"
    if not path.exists():
        return {}
    raw = read_json(path)
    skills = raw.get("skills", raw) if isinstance(raw, dict) else raw
    return {str(k): v for k, v in (skills or {}).items() if isinstance(v, dict)} if isinstance(skills, dict) else {}


def _is_recovery_experience(record: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(record, dict):
        return False
    stats = record.get("stats") if isinstance(record.get("stats"), dict) else {}
    return bool(record.get("recovery_status") or stats.get("recovered_from_failure") or stats.get("recovery_status"))


def _is_recovery_skill(skill_id: str, skill: Optional[Dict[str, Any]], experiences: Dict[str, Dict[str, Any]]) -> bool:
    skill_id_text = str(skill_id).lower()
    if "recovery" in skill_id_text:
        return True
    if not isinstance(skill, dict):
        return False
    if (
        skill.get("recovery_status")
        or "recovery" in str(skill.get("skill_type", "")).lower()
        or "recovery" in str(skill.get("name", "")).lower()
    ):
        return True
    return any(_is_recovery_experience(experiences.get(str(exp_id))) for exp_id in skill.get("linked_experience_ids") or [])


def _mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else None


def _empty_action_usage_stats() -> Dict[str, Any]:
    return {
        "sample_count": 0,
        "selected_skill_samples": 0,
        "recovery_skill_samples": 0,
        "unverified_recovery_skill_samples": 0,
        "recovery_probe_samples": 0,
        "mandatory_plan_required_samples": 0,
        "memory_plan_available_samples": 0,
        "direct_plan_samples": 0,
        "action_memory_samples": 0,
        "selected_skill_counts": Counter(),
        "recovery_skill_counts": Counter(),
        "action_experience_counts": Counter(),
        "direct_experience_counts": Counter(),
        "recovery_probe_skill_counts": Counter(),
        "grouped_scores": defaultdict(list),
        "grouped_passes": defaultdict(list),
    }


def _record_signal_outcome(stats: Dict[str, Any], key: str, pass_k: float, avg_k: float) -> None:
    stats["grouped_scores"][key].append(avg_k)
    stats["grouped_passes"][key].append(pass_k)


def _accumulate_action_usage_sample(
    stats: Dict[str, Any],
    usage: Dict[str, Any],
    *,
    pass_k: float,
    avg_k: float,
    skills: Dict[str, Dict[str, Any]],
    experiences: Dict[str, Dict[str, Any]],
) -> None:
    stats["sample_count"] += 1
    selected_skill_ids = [str(sid) for sid in usage.get("selected_skill_ids") or []]
    action_experience_ids = [str(exp_id) for exp_id in usage.get("action_experience_ids") or []]
    direct_plan_ids = [str(exp_id) for exp_id in usage.get("direct_tool_plan_experience_ids") or []]
    retrieval_info = usage.get("skill_retrieval_info") if isinstance(usage.get("skill_retrieval_info"), dict) else {}
    probe_ids = [str(sid) for sid in retrieval_info.get("recovery_probe_extra_skill_ids") or []]

    if selected_skill_ids:
        stats["selected_skill_samples"] += 1
        stats["selected_skill_counts"].update(selected_skill_ids)
        _record_signal_outcome(stats, "selected_skill", pass_k, avg_k)
    else:
        _record_signal_outcome(stats, "no_selected_skill", pass_k, avg_k)

    recovery_selected = [
        sid for sid in selected_skill_ids
        if _is_recovery_skill(sid, skills.get(sid), experiences)
    ]
    if recovery_selected:
        stats["recovery_skill_samples"] += 1
        stats["recovery_skill_counts"].update(recovery_selected)
        _record_signal_outcome(stats, "recovery_skill", pass_k, avg_k)
    if any(str((skills.get(sid) or {}).get("recovery_status", "")) == "candidate_unverified" for sid in recovery_selected):
        stats["unverified_recovery_skill_samples"] += 1
        _record_signal_outcome(stats, "unverified_recovery_skill", pass_k, avg_k)

    if probe_ids:
        stats["recovery_probe_samples"] += 1
        stats["recovery_probe_skill_counts"].update(probe_ids)
    if usage.get("mandatory_tool_plan_required"):
        stats["mandatory_plan_required_samples"] += 1
    if usage.get("memory_tool_plan_available"):
        stats["memory_plan_available_samples"] += 1
        _record_signal_outcome(stats, "memory_tool_plan", pass_k, avg_k)
    if direct_plan_ids:
        stats["direct_plan_samples"] += 1
        stats["direct_experience_counts"].update(direct_plan_ids)
        _record_signal_outcome(stats, "direct_tool_plan", pass_k, avg_k)
    if action_experience_ids:
        stats["action_memory_samples"] += 1
        stats["action_experience_counts"].update(action_experience_ids)
        _record_signal_outcome(stats, "action_memory", pass_k, avg_k)


def _outcome_by_signal(stats: Dict[str, Any], k: int) -> Dict[str, Any]:
    return {
        key: {
            "num_samples": len(values),
            f"average@{k}": _mean(values),
            f"pass@{k}": _mean(stats["grouped_passes"].get(key, [])),
        }
        for key, values in sorted(stats["grouped_scores"].items())
    }


def _action_usage_result(stats: Dict[str, Any], k: int) -> Dict[str, Any]:
    sample_count = stats["sample_count"]

    def ratio(count: int, denom: int | None = None) -> float:
        base = sample_count if denom is None else denom
        return count / base if base else 0.0

    return {
        "samples": sample_count,
        "selected_skill_samples": stats["selected_skill_samples"],
        "selected_skill_ratio": ratio(stats["selected_skill_samples"]),
        "recovery_skill_samples": stats["recovery_skill_samples"],
        "recovery_skill_ratio": ratio(stats["recovery_skill_samples"]),
        "unverified_recovery_skill_samples": stats["unverified_recovery_skill_samples"],
        "unverified_recovery_skill_ratio": ratio(stats["unverified_recovery_skill_samples"]),
        "recovery_probe_samples": stats["recovery_probe_samples"],
        "recovery_probe_ratio": ratio(stats["recovery_probe_samples"]),
        "mandatory_tool_plan_required_samples": stats["mandatory_plan_required_samples"],
        "memory_tool_plan_available_samples": stats["memory_plan_available_samples"],
        "memory_tool_plan_available_ratio": ratio(stats["memory_plan_available_samples"]),
        "memory_tool_plan_coverage": ratio(stats["memory_plan_available_samples"], stats["mandatory_plan_required_samples"]),
        "direct_tool_plan_samples": stats["direct_plan_samples"],
        "direct_tool_plan_ratio": ratio(stats["direct_plan_samples"]),
        "action_memory_samples": stats["action_memory_samples"],
        "action_memory_ratio": ratio(stats["action_memory_samples"]),
        "top_selected_skill_ids": stats["selected_skill_counts"].most_common(10),
        "top_recovery_skill_ids": stats["recovery_skill_counts"].most_common(10),
        "top_recovery_probe_skill_ids": stats["recovery_probe_skill_counts"].most_common(10),
        "top_action_experience_ids": stats["action_experience_counts"].most_common(10),
        "top_direct_tool_plan_experience_ids": stats["direct_experience_counts"].most_common(10),
        "outcome_by_signal": _outcome_by_signal(stats, k),
    }


def memory_action_usage(metrics_dir: Path, memory_root: Path, k: int) -> Dict[str, Any]:
    """Summarize whether memory produced executable action signals.

    This is intentionally diagnostic only: it reads per-sample
    tt_memory_usage.json files and does not influence training or validation.
    """
    experiences = _load_experiences(memory_root)
    skills = _load_skills(memory_root)
    stats = _empty_action_usage_stats()

    for usage_path in sorted(metrics_dir.glob(f"*/{MEMORY_USAGE_FILENAME}")):
        usage = read_json(usage_path)
        if not isinstance(usage, dict):
            raise ValueError(f"{usage_path} must contain a JSON object.")
        pass_k, avg_k = sample_metric(usage_path.parent, k)
        _accumulate_action_usage_sample(
            stats,
            usage,
            pass_k=pass_k,
            avg_k=avg_k,
            skills=skills,
            experiences=experiences,
        )
    return _action_usage_result(stats, k)


def sample_metric(sample_dir: Path, k: int) -> Tuple[float, float]:
    metrics_path = sample_dir / SAMPLE_SUMMARY_FILENAME
    if not metrics_path.exists():
        return 0.0, 0.0
    data = read_json(metrics_path)
    pass_k = float(data.get(f"pass@{k}", 0.0) or 0.0)
    avg_k = float(data.get(f"average@{k}", 0.0) or 0.0)
    return pass_k, avg_k
