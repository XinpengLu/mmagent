"""Sample- and dataset-level metrics for benchmark runs."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from skill import load_skill_index
from utils.artifacts import (
    DATASET_SUMMARY_FILENAME,
    MEMORY_USAGE_FILENAME,
    RESULTS_FILENAME,
    ROLLOUT_METRICS_FILENAME,
    SAMPLE_SUMMARY_FILENAME,
    TRAJECTORY_FILENAME,
    iter_jsonl,
    read_json,
    write_json,
)
from utils.transient import result_has_transient_failure


def _save_results(results, output_dir):
    """Save rollout results to results.jsonl."""
    output_file_path = os.path.join(output_dir, RESULTS_FILENAME)
    with open(output_file_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Results saved to {output_file_path}")


def compute_and_save_sample_summary(question_id, sample_rollout_results, rollouts_per_sample, sample_dir):
    """Compute sample-level pass@k, average@k, and rollout diagnostics."""
    if not sample_rollout_results:
        return None

    sorted_results = (
        sorted(sample_rollout_results, key=lambda r: r.get("_rollout_idx", 0))
        if "_rollout_idx" in sample_rollout_results[0]
        else sample_rollout_results
    )
    invalid_rollout_flags = [result_has_transient_failure(r) for r in sorted_results]
    invalid_rollout_indices = [
        int(r.get("_rollout_idx", idx))
        for idx, (r, is_invalid) in enumerate(zip(sorted_results, invalid_rollout_flags))
        if is_invalid
    ]
    accuracies = [r["accuracy_score"] for r in sorted_results]
    sample_summary = {
        "question_id": question_id,
        "num_rollouts": len(sorted_results),
        "accuracies": accuracies,
        "has_transient_failure": bool(invalid_rollout_indices),
        "invalid_rollout_count": len(invalid_rollout_indices),
        "invalid_rollout_indices": invalid_rollout_indices,
    }

    numeric_fields = [
        "runtime_seconds",
        "api_elapsed_seconds",
        "api_call_count",
        "api_prompt_tokens",
        "api_completion_tokens",
        "api_total_tokens",
        "estimated_context_tokens",
        "turn_count",
        "tool_call_count",
    ]
    for field in numeric_fields:
        values = [float(r.get(field) or 0.0) for r in sorted_results if r.get(field) is not None]
        if values:
            sample_summary[f"{field}_mean"] = sum(values) / len(values)
            sample_summary[f"{field}_sum"] = sum(values)

    sample_summary["sampling_temperatures"] = [
        r.get("sampling_temperature") for r in sorted_results if r.get("sampling_temperature") is not None
    ]
    for k in range(1, len(accuracies) + 1):
        acc_at_k = accuracies[:k]
        sample_summary[f"pass@{k}"] = 1.0 if any(acc == 1.0 for acc in acc_at_k) else 0.0
        sample_summary[f"average@{k}"] = sum(acc_at_k) / k

    write_json(os.path.join(sample_dir, SAMPLE_SUMMARY_FILENAME), sample_summary)
    if invalid_rollout_indices:
        print(
            f"  \nSample {question_id}: contains invalid rollout(s) "
            f"{invalid_rollout_indices}; rerun before scoring."
        )
    print(
        f"  \nSample {question_id}: "
        f"pass@{rollouts_per_sample}={sample_summary[f'pass@{rollouts_per_sample}']:.2f}, "
        f"average@{rollouts_per_sample}={sample_summary[f'average@{rollouts_per_sample}']:.4f}"
    )
    return sample_summary


def _summary_group_key(sample: Dict[str, Any]) -> str:
    for field in (
        "task",
        "domain",
        "subdomain",
        "category",
        "subtask",
        "prompt_category",
        "eval_focus",
        "difficulty",
        "level",
        "data_source",
    ):
        value = sample.get(field)
        if value not in (None, ""):
            return f"{field}={value}"
    return "all"


def _aggregate_sample_summaries(sample_summaries: List[Dict[str, Any]], rollouts_per_sample: int) -> Tuple[Dict[str, Any], Dict[str, int]]:
    k_metrics: Dict[str, Any] = {}
    for k in range(1, rollouts_per_sample + 1):
        pass_values = [s[f"pass@{k}"] for s in sample_summaries if f"pass@{k}" in s]
        avg_values = [s[f"average@{k}"] for s in sample_summaries if f"average@{k}" in s]
        pass_at_k = round(sum(pass_values) / len(pass_values), 4) if pass_values else 0.0
        average_at_k = round(sum(avg_values) / len(avg_values), 4) if avg_values else 0.0
        k_metrics[str(k)] = {
            "pass_at_k": pass_at_k,
            "average_at_k": average_at_k,
            "stability_gap": round(pass_at_k - average_at_k, 4),
        }
    final_k = rollouts_per_sample
    outcome_counts = {
        "all_wrong": sum(1 for s in sample_summaries if float(s.get(f"pass@{final_k}", 0.0) or 0.0) <= 0.0),
        "all_correct": sum(1 for s in sample_summaries if float(s.get(f"average@{final_k}", 0.0) or 0.0) >= 1.0),
        "partial": sum(1 for s in sample_summaries if 0.0 < float(s.get(f"average@{final_k}", 0.0) or 0.0) < 1.0),
    }
    return k_metrics, outcome_counts


def _summarize_tool_usage(output_dir: str) -> Dict[str, Any]:
    tool_calls: Counter = Counter()
    rollouts = 0
    tool_rollouts = 0
    calls_per_rollout: List[int] = []
    accuracy_with_tool: List[float] = []
    accuracy_without_tool: List[float] = []
    for root, _, files in os.walk(output_dir):
        if TRAJECTORY_FILENAME not in files:
            continue
        rollouts += 1
        call_count = 0
        for item in iter_jsonl(os.path.join(root, TRAJECTORY_FILENAME)):
            call = item.get("tool_call")
            if isinstance(call, dict):
                tool_calls[str(call.get("tool_name") or call.get("name") or "unknown")] += 1
                call_count += 1
        metrics_path = os.path.join(root, ROLLOUT_METRICS_FILENAME)
        metrics = read_json(metrics_path) if os.path.exists(metrics_path) else {}
        accuracy = float(metrics.get("accuracy_score", 0.0) or 0.0)
        calls_per_rollout.append(call_count)
        if call_count:
            tool_rollouts += 1
            accuracy_with_tool.append(accuracy)
        else:
            accuracy_without_tool.append(accuracy)
    return {
        "rollouts": rollouts,
        "tool_rollouts": tool_rollouts,
        "tool_rollout_rate": round(tool_rollouts / rollouts, 4) if rollouts else 0.0,
        "avg_tool_calls_per_rollout": round(sum(calls_per_rollout) / len(calls_per_rollout), 4) if calls_per_rollout else 0.0,
        "tool_call_counts": dict(tool_calls.most_common()),
        "accuracy_with_tool": round(sum(accuracy_with_tool) / len(accuracy_with_tool), 4) if accuracy_with_tool else None,
        "accuracy_without_tool": round(sum(accuracy_without_tool) / len(accuracy_without_tool), 4) if accuracy_without_tool else None,
    }


def _summarize_memory_usage(output_dir: str) -> Dict[str, Any]:
    skill_counts: Counter = Counter()
    exp_counts: Counter = Counter()
    samples = 0
    with_skill = 0
    with_exp = 0
    mandatory = 0
    dropped = 0
    for root, _, files in os.walk(output_dir):
        if MEMORY_USAGE_FILENAME not in files:
            continue
        usage = read_json(os.path.join(root, MEMORY_USAGE_FILENAME))
        samples += 1
        selected_skills = usage.get("selected_skill_ids") or []
        selected_exps = usage.get("retrieved_experience_ids") or []
        if selected_skills:
            with_skill += 1
            skill_counts.update(str(x) for x in selected_skills)
        if selected_exps:
            with_exp += 1
            exp_counts.update(str(x) for x in selected_exps)
        if usage.get("mandatory_tool_plan_required"):
            mandatory += 1
        dropped += len(usage.get("quality_dropped_experience_ids") or [])
    return {
        "samples_with_memory_record": samples,
        "skill_selection_rate": round(with_skill / samples, 4) if samples else 0.0,
        "experience_selection_rate": round(with_exp / samples, 4) if samples else 0.0,
        "mandatory_tool_plan_rate": round(mandatory / samples, 4) if samples else 0.0,
        "quality_dropped_experience_count": dropped,
        "top_selected_skills": skill_counts.most_common(20),
        "top_selected_experiences": exp_counts.most_common(20),
    }


def _summarize_memory_bank(args) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    exp_path = getattr(args, "experience_library", None)
    if exp_path and os.path.exists(exp_path):
        raw = read_json(exp_path)
        exps = raw.get("experiences", raw) if isinstance(raw, dict) else {}
        if isinstance(exps, dict):
            values = [v for v in exps.values() if isinstance(v, dict)]
            stats["experience_count"] = len(values)
            stats["experience_kind_counts"] = dict(Counter(v.get("kind", "unknown") for v in values))
            stats["experience_with_tool_sequence"] = sum(bool(v.get("tool_sequence")) for v in values)
            stats["experience_with_validation_checks"] = sum(bool(v.get("validation_checks")) for v in values)
            stats["experience_with_fallback_plan"] = sum(bool(v.get("fallback_plan")) for v in values)
    skill_path = getattr(args, "skill_library", None)
    if skill_path:
        index = load_skill_index(skill_path)
        skills = index.get("skills", {}) if isinstance(index, dict) else {}
        values = [v for v in skills.values() if isinstance(v, dict)]
        stats["skill_count"] = len(values)
        stats["skill_type_counts"] = dict(Counter(v.get("skill_type", "unknown") for v in values))
        stats["skills_with_validation_quality"] = sum(v.get("validation_quality") is not None for v in values)
        stats["disabled_skill_count"] = sum(bool(v.get("disabled")) for v in values)
    return stats


def compute_dataset_summary(data, args):
    """Compute dataset-level pass@k, stability, task, tool, and memory metrics."""
    sample_summaries = []
    sample_summary_by_id: Dict[str, Dict[str, Any]] = {}
    sample_by_id: Dict[str, Dict[str, Any]] = {}
    for idx, sample in enumerate(data):
        question_id = sample.get("doc_id", sample.get("question_id", f"sample_{idx}"))
        summary_path = os.path.join(args.output_dir, question_id, SAMPLE_SUMMARY_FILENAME)
        if os.path.exists(summary_path):
            summary = read_json(summary_path)
            sample_summaries.append(summary)
            sample_summary_by_id[question_id] = summary
            sample_by_id[question_id] = sample
    if not sample_summaries:
        return None

    k_metrics, outcome_counts = _aggregate_sample_summaries(sample_summaries, args.rollouts_per_sample)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for question_id, summary in sample_summary_by_id.items():
        grouped[_summary_group_key(sample_by_id.get(question_id, {}))].append(summary)
    per_group_metrics = {}
    for group, items in sorted(grouped.items()):
        group_k_metrics, group_outcomes = _aggregate_sample_summaries(items, args.rollouts_per_sample)
        per_group_metrics[group] = {
            "num_samples": len(items),
            "k_metrics": group_k_metrics,
            "outcome_counts": group_outcomes,
        }

    dataset_summary = {
        "num_samples": len(sample_summaries),
        "num_rollouts": args.rollouts_per_sample,
        "k_metrics": k_metrics,
        "outcome_counts": outcome_counts,
        "per_group_metrics": per_group_metrics,
        "tool_usage": _summarize_tool_usage(args.output_dir),
        "memory_usage": _summarize_memory_usage(args.output_dir),
        "memory_bank": _summarize_memory_bank(args),
        "config": {
            "system_prompt_key": getattr(args, "system_prompt_key", None),
            "experience_retrieval": bool(getattr(args, "experience_retrieval", False)),
            "experience_retrieval_decomposition": bool(getattr(args, "experience_retrieval_decomposition", False)),
            "experience_retrieval_rewrite": bool(getattr(args, "experience_retrieval_rewrite", False)),
            "skill_inference": bool(getattr(args, "skill_inference", False)),
            "memory_rerank_top_skills": getattr(args, "memory_rerank_top_skills", None),
            "memory_rerank_top_experiences": getattr(args, "memory_rerank_top_experiences", None),
        },
    }
    write_json(os.path.join(args.output_dir, DATASET_SUMMARY_FILENAME), dataset_summary)
    print(f"Total samples: {dataset_summary['num_samples']}")
    print(f"Rollouts per sample: {dataset_summary['num_rollouts']}\n")
    for k in range(1, args.rollouts_per_sample + 1):
        metrics = dataset_summary["k_metrics"][str(k)]
        print(f"k={k}:")
        print(f"  Pass@{k}:    {metrics['pass_at_k']:.4f}")
        print(f"  Average@{k}: {metrics['average_at_k']:.4f}")
        print(f"  Gap@{k}:     {metrics['stability_gap']:.4f}")
    print("=" * 80)
    return dataset_summary
