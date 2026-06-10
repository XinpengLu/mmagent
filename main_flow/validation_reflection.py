"""Validation-set reflection for memory improvement.

This module is intentionally small: it reuses the existing supervised
correction prompt to turn low-scoring validation trajectories into unverified
recovery candidates. The validation metric is still recorded before these
candidates affect later stages.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from experience import ExperienceLLM, batch_merge, load_experiences, save_library
from experience.critique import supervised_correction_experiences
from skill import cluster_experiences_to_skills

from validation_tools.metrics import sample_metric
from validation_tools.tasks import task_of_sample
from utils.artifacts import MEMORY_USAGE_FILENAME, ROLLOUT_METRICS_FILENAME, TRAJECTORY_FILENAME
from utils.artifacts import read_json, write_json
from configs.runtime import runtime_setting


def _sample_id(sample: Dict[str, Any]) -> str:
    return str(sample.get("doc_id") or sample.get("question_id") or sample.get("id") or "").strip()


def _load_samples_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return {_sample_id(item): item for item in data if isinstance(item, dict) and _sample_id(item)}


def _question(sample: Dict[str, Any]) -> str:
    return str(sample.get("problem") or sample.get("question") or sample.get("query") or sample.get("prompt") or "")


def _groundtruth(sample: Dict[str, Any]) -> str:
    value = sample.get("solution")
    if value is None:
        value = sample.get("answer") or sample.get("ground_truth") or sample.get("label")
    return str(value or "")


def _clean_text(text: str, limit: int = 900) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "[tool_image]", str(text or ""))
    text = re.sub(r"array\(\[.*?shape=\([^)]+\), dtype=[^)]+\)", "[large_array]", text, flags=re.S)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _compact_rollout_summary(rollout_dir: Path) -> str:
    traj = rollout_dir / TRAJECTORY_FILENAME
    metrics = {}
    if (rollout_dir / ROLLOUT_METRICS_FILENAME).exists():
        metrics = read_json(rollout_dir / ROLLOUT_METRICS_FILENAME)

    snippets: List[str] = []
    final_text = ""
    tool_count = 0
    if traj.exists():
        for line in traj.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if "text_output" in item:
                text = str(item.get("text_output") or "")
                if "<answer>" in text or "Final" in text:
                    final_text = _clean_text(text, 700)
            tool_call = item.get("tool_call") if isinstance(item, dict) else None
            if isinstance(tool_call, dict):
                tool_count += 1
                params = tool_call.get("parameters") or {}
                code = _clean_text(params.get("code") or "", 900)
                result = _clean_text(tool_call.get("result") or "", 500)
                snippets.append(f"Tool {tool_count}: code={code}; result={result}")

    accuracy = metrics.get("accuracy_score", metrics.get("accuracy", "unknown"))
    selected = snippets[-4:]
    parts = [
        f"accuracy={accuracy}",
        f"tool_calls={tool_count}",
        f"final={final_text or '[no final answer found]'}",
    ]
    parts.extend(selected)
    return "\n".join(parts)


def _used_experience_context(memory_root: Path, sample_dir: Path, limit: int = 8) -> Dict[str, Dict[str, Any]]:
    """Return the retrieved experiences that were present during this failed validation sample."""
    usage_path = sample_dir / MEMORY_USAGE_FILENAME
    exp_path = memory_root / "experiences.json"
    if not usage_path.exists() or not exp_path.exists():
        return {}
    usage = read_json(usage_path)
    raw = read_json(exp_path)

    exp_lib = raw.get("experiences", raw) if isinstance(raw, dict) else {}
    if not isinstance(exp_lib, dict):
        return {}

    ordered_ids: List[str] = []
    for field in (
        "action_experience_ids",
        "direct_tool_plan_experience_ids",
        "linked_experience_ids",
        "selected_experience_ids",
        "retrieved_experience_ids",
        "direct_retrieved_experience_ids",
        "original_retrieved_experience_ids",
    ):
        for exp_id in usage.get(field, []) or []:
            exp_id = str(exp_id)
            if exp_id not in ordered_ids:
                ordered_ids.append(exp_id)

    out: Dict[str, Dict[str, Any]] = {}
    for exp_id in ordered_ids:
        record = exp_lib.get(exp_id)
        if isinstance(record, dict):
            out[exp_id] = record
        if len(out) >= limit:
            break
    return out


def _previous_reflection_samples(output_json: Path, stage: str) -> set[str]:
    if not output_json.exists():
        return set()
    previous = read_json(output_json)
    if str(previous.get("stage")) != str(stage):
        return set()
    return {
        str(row.get("sample_id"))
        for row in previous.get("samples", []) or []
        if isinstance(row, dict)
        and (int(row.get("operations", 0) or 0) > 0 or row.get("skipped_existing_reflection"))
    }


def _existing_reflection_samples(memory_root: Path, stage: str) -> set[str]:
    exp_path = memory_root / "experiences.json"
    if not exp_path.exists():
        return set()
    raw = read_json(exp_path)
    exp_lib = raw.get("experiences", raw) if isinstance(raw, dict) else {}
    if not isinstance(exp_lib, dict):
        return set()
    samples = set()
    for record in exp_lib.values():
        if not isinstance(record, dict):
            continue
        if str(record.get("source_validation_stage", "")) == str(stage):
            sample_id = str(record.get("source_validation_sample_id", "")).strip()
            if sample_id:
                samples.add(sample_id)
    return samples


def _candidate_samples(
    metrics_dir: Path,
    data_by_id: Dict[str, Dict[str, Any]],
    *,
    k: int,
    max_samples: int,
    min_average: float,
) -> List[Tuple[Tuple[bool, float], Path, Dict[str, Any]]]:
    rows: List[Tuple[Tuple[bool, float], Path, Dict[str, Any]]] = []
    for sample_dir in metrics_dir.iterdir() if metrics_dir.exists() else []:
        if not sample_dir.is_dir():
            continue
        sample = data_by_id.get(sample_dir.name)
        if not sample:
            continue
        pass_k, avg_k = sample_metric(sample_dir, k)
        if pass_k > 0 and avg_k >= min_average:
            continue
        reflection_need = (pass_k <= 0, -float(avg_k or 0.0))
        rows.append((reflection_need, sample_dir, sample))
    rows.sort(key=lambda item: item[0], reverse=True)
    return rows[: max(0, max_samples)]


def _empty_reflection_result(stage: str) -> Dict[str, Any]:
    return {
        "stage": str(stage),
        "candidates": 0,
        "operations": 0,
        "write_back": False,
        "samples": [],
    }


def _finish_reflection(output_json: Path, result: Dict[str, Any]) -> None:
    write_json(output_json, result, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _pending_reflection_samples(
    candidates: List[Tuple[Tuple[bool, float], Path, Dict[str, Any]]],
    reflected_samples: set[str],
    result: Dict[str, Any],
) -> List[Tuple[Tuple[bool, float], Path, Dict[str, Any]]]:
    pending = []
    for candidate in candidates:
        _, sample_dir, _ = candidate
        if sample_dir.name in reflected_samples:
            result["samples"].append(
                {
                    "sample_id": sample_dir.name,
                    "operations": 0,
                    "skipped_existing_reflection": True,
                }
            )
        else:
            pending.append(candidate)
    return pending


def _rollout_summaries(sample_dir: Path) -> Dict[str, str]:
    summaries = {}
    for rollout_dir in sorted(sample_dir.glob("rollout_*")):
        if rollout_dir.is_dir():
            summaries[rollout_dir.name] = _compact_rollout_summary(rollout_dir)
    return summaries


def _validation_reflection_tags(sample: Dict[str, Any]) -> List[str]:
    tags = ["validation_reflection"]
    task = task_of_sample(sample)
    if task and task != "unknown":
        tags.append(f"task_{task}")
    data_source = sample.get("data_source")
    if data_source:
        tags.append(f"data_source_{str(data_source).strip().lower().replace('-', '_')}")
    return tags


def _tag_validation_operations(ops: List[dict], sample: Dict[str, Any], sample_dir: Path, stage: str) -> None:
    extra_tags = _validation_reflection_tags(sample)
    for op in ops:
        op["source_validation_stage"] = str(stage)
        op["source_validation_sample_id"] = sample_dir.name
        tags = list(op.get("task_tags") or [])
        for tag in extra_tags:
            if tag not in tags:
                tags.append(tag)
        op["task_tags"] = tags


def _reflect_one_sample(
    *,
    sample_dir: Path,
    sample: Dict[str, Any],
    memory_root: Path,
    stage: str,
    llm: ExperienceLLM,
) -> List[dict]:
    summaries = _rollout_summaries(sample_dir)
    if not summaries:
        return []

    ops = supervised_correction_experiences(
        _question(sample),
        _groundtruth(sample),
        summaries,
        llm,
        debug_dir=str(sample_dir / "validation_reflection"),
        used_experiences=_used_experience_context(memory_root, sample_dir),
    )
    if ops:
        _tag_validation_operations(ops, sample, sample_dir, stage)
    return ops


def _write_reflection_memory(
    *,
    memory_root: Path,
    metrics_dir: Path,
    skill_library: Path,
    operations: List[dict],
) -> None:
    exp_path = memory_root / "experiences.json"
    merged = batch_merge(
        load_experiences(str(exp_path)),
        operations,
        ExperienceLLM(),
        debug_dir=str(metrics_dir / "validation_reflection_merge"),
        experience_limit=int(runtime_setting("experience.max_items", 90)),
    )
    memory_root.mkdir(parents=True, exist_ok=True)
    save_library(str(exp_path), merged)

    skill_args = SimpleNamespace(
        skill_cluster_min_experiences=int(runtime_setting("skill.cluster.min_experiences", 2)),
        skill_cluster_max_clusters=int(runtime_setting("skill.cluster.max_clusters", 12)),
        skill_max_length=1000,
    )
    merged = cluster_experiences_to_skills(merged, str(skill_library), ExperienceLLM(), skill_args)
    save_library(str(exp_path), merged)


def reflect_validation_failures(
    *,
    metrics_dir: Path,
    memory_root: Path,
    data_file: Path,
    stage: str,
    output_json: Path,
    skill_library: Path,
    k: int,
    max_samples: int,
    min_average: float,
) -> None:
    data_by_id = _load_samples_by_id(data_file)
    result = _empty_reflection_result(stage)

    if not data_by_id or not metrics_dir.exists():
        _finish_reflection(output_json, result)
        return

    reflected_samples = _previous_reflection_samples(output_json, str(stage))
    reflected_samples |= _existing_reflection_samples(memory_root, str(stage))
    candidates = _candidate_samples(
        metrics_dir,
        data_by_id,
        k=k,
        max_samples=max_samples,
        min_average=min_average,
    )
    pending = _pending_reflection_samples(candidates, reflected_samples, result)
    if not pending:
        result["candidates"] = len(result["samples"])
        _finish_reflection(output_json, result)
        return

    llm = ExperienceLLM()
    operations: List[dict] = []
    for _, sample_dir, sample in pending:
        ops = _reflect_one_sample(
            sample_dir=sample_dir,
            sample=sample,
            memory_root=memory_root,
            stage=str(stage),
            llm=llm,
        )
        if ops:
            operations.extend(ops)
        result["samples"].append(
            {
                "sample_id": sample_dir.name,
                "operations": len(ops),
            }
        )

    result["candidates"] = len(result["samples"])
    result["operations"] = len(operations)
    result["write_back"] = bool(operations)
    if operations:
        _write_reflection_memory(
            memory_root=memory_root,
            metrics_dir=metrics_dir,
            skill_library=skill_library,
            operations=operations,
        )

    _finish_reflection(output_json, result)
