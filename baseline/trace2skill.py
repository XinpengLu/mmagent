"""Trace2Skill-adapted baseline built on the existing rollout runner."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from engine.api_caller import call_vision_api
from main_flow.metrics import compute_and_save_sample_summary
from utils.artifacts import read_json, write_json
from utils.transient import result_has_transient_failure

from baseline.reflexion import (
    _load_completed_rollouts,
    _question_id,
    _rollout_dir,
    _run_rollout,
    _sample_ground_truth,
    _sample_question,
    _trajectory_text,
)


TRACE2SKILL_MEMORY_SYSTEM_PROMPT = (
    "You are a Trace2Skill trajectory-analysis agent for multimodal tool-using agents. "
    "Analyze the trajectory evidence and extract reusable memory items that can improve "
    "future agent behavior. Keep every item general, evidence-driven, and free of "
    "sample-specific answers."
)

TRACE2SKILL_CONSOLIDATION_SYSTEM_PROMPT = (
    "You are consolidating Trace2Skill memory items into a compact SKILL.md document for "
    "a multimodal tool-using agent. Merge duplicates, resolve conflicts, and keep only "
    "actionable guidance that transfers across tasks."
)


def _chat_text(model_name: str, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    response = call_vision_api(
        model_name,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        {"temperature": 0.2, "top_p": 1.0, "max_tokens": max_tokens},
        tools=None,
    )
    if isinstance(response, dict):
        return str(response.get("content") or "").strip()
    return str(response or "").strip()


def _trace2skill_root(args) -> Path:
    return Path(args.train_output_dir)


def _local_lessons_path(args) -> Path:
    return _trace2skill_root(args) / "trace2skill_memory_items.json"


def _guide_path(args) -> Path:
    return _trace2skill_root(args) / "SKILL.md"


def _memory_item_prompt(sample: dict[str, Any], result: dict[str, Any], rollout_dir: Path) -> str:
    outcome = "success" if float(result.get("accuracy_score", 0.0) or 0.0) >= 1.0 else "failure"
    item_type = "Success Memory Items" if outcome == "success" else "Failure Memory Items"
    return (
        f"Task outcome: {outcome}\n\n"
        f"Question:\n{_sample_question(sample)}\n\n"
        f"Ground truth:\n{_sample_ground_truth(sample)}\n\n"
        f"Final answer:\n{result.get('final_answer', '')}\n\n"
        f"Judge analysis:\n{result.get('trajectory_analysis', '')}\n\n"
        f"Trajectory excerpt:\n{_trajectory_text(rollout_dir)}\n\n"
        f"Produce no more than 3 {item_type}. Use this exact markdown format for each item:\n\n"
        f"# {'Success' if outcome == 'success' else 'Failure'} Memory Item <i>\n"
        "## Title\n<short title>\n\n"
        "## Description\n<one-sentence summary>\n\n"
        "## Content\n<1-3 sentences with the reusable lesson>\n"
    )


def _parse_memory_items(text: str, outcome: str) -> list[dict[str, str]]:
    marker = "Success Memory Item" if outcome == "success" else "Failure Memory Item"
    chunks = re.split(rf"(?=^#\s+{re.escape(marker)}\s+\d+)", text, flags=re.M)
    items = []
    for chunk in chunks:
        if marker not in chunk:
            continue
        title = _extract_section(chunk, "Title")
        description = _extract_section(chunk, "Description")
        content = _extract_section(chunk, "Content")
        if content:
            items.append({"type": outcome, "title": title, "description": description, "content": content})
    return items


def _extract_section(text: str, heading: str) -> str:
    match = re.search(rf"##\s+{re.escape(heading)}\s*\n(.+?)(?=\n##\s+|\Z)", text, flags=re.S)
    return match.group(1).strip() if match else ""


def _extract_memory_items(sample: dict[str, Any], result: dict[str, Any], args, rollout_dir: Path) -> list[dict[str, str]]:
    outcome = "success" if float(result.get("accuracy_score", 0.0) or 0.0) >= 1.0 else "failure"
    text = _chat_text(
        args.model_name,
        TRACE2SKILL_MEMORY_SYSTEM_PROMPT,
        _memory_item_prompt(sample, result, rollout_dir),
        max_tokens=int(args.trace2skill_lesson_max_tokens),
    )
    return _parse_memory_items(text, outcome)


def _run_trace2skill_training_sample(
    sample_idx: int,
    sample: dict[str, Any],
    total_samples: int,
    args,
    sampling_params: dict[str, Any],
) -> tuple[int, dict[str, Any] | None]:
    question_id = _question_id(sample, sample_idx)
    print(f"[Trace2Skill] Training sample {sample_idx + 1}/{total_samples} {question_id}: extracting memory items")
    result = _run_rollout(sample, args, sampling_params, rollout_idx=0)
    if not result or result_has_transient_failure(result):
        return sample_idx, None

    rollout_dir = _rollout_dir(args.output_dir, question_id, 0, args.rollouts_per_sample)
    memory_items = _extract_memory_items(sample, result, args, rollout_dir)
    if not memory_items:
        return sample_idx, None

    return sample_idx, {
        "source_question_id": question_id,
        "source_score": result.get("accuracy_score", 0.0),
        "outcome": "success" if float(result.get("accuracy_score", 0.0) or 0.0) >= 1.0 else "failure",
        "memory_items": memory_items,
    }


def _collect_local_lessons(config, args) -> list[dict[str, Any]]:
    train_file = Path(args.input_file).parent / "train.json"
    lesson_path = _local_lessons_path(args)
    if lesson_path.exists():
        return list(read_json(lesson_path).get("memory_items", []))
    if getattr(args, "reuse_train_benchmark", ""):
        raise FileNotFoundError(
            f"Missing reusable Trace2Skill memory items for {args.benchmark}: {lesson_path}. "
            f"Run {args.reuse_train_benchmark} before {args.benchmark}."
        )

    train_data = json.loads(train_file.read_text(encoding="utf-8"))[: int(args.trace2skill_train_samples)]
    train_args = SimpleNamespace(**vars(args))
    train_args.phase = "train"
    train_args.input_file = str(train_file)
    train_args.output_dir = str(_trace2skill_root(args))
    train_args.rollouts_per_sample = 1
    train_args.num_workers = int(args.num_workers)
    train_args._sample_system_prompt = config.base_system_prompt

    max_workers = max(1, min(int(args.num_workers), len(train_data)))
    print(f"[Trace2Skill] Training {len(train_data)} samples with {max_workers} worker(s).")
    memory_outputs: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_trace2skill_training_sample,
                sample_idx,
                sample,
                len(train_data),
                SimpleNamespace(**vars(train_args)),
                config.sampling_params,
            )
            for sample_idx, sample in enumerate(train_data)
        ]
        for future in as_completed(futures):
            sample_idx, record = future.result()
            if record is not None:
                memory_outputs[sample_idx] = record

    memory_records = [memory_outputs[idx] for idx in sorted(memory_outputs)]
    write_json(lesson_path, {"memory_items": memory_records}, indent=2)
    return memory_records


def _consolidation_prompt(lesson_records: list[dict[str, Any]]) -> str:
    blocks = []
    for record in lesson_records:
        items = "\n".join(
            f"- [{item['type']}] {item.get('title', '')}: {item.get('content', '')}"
            for item in record["memory_items"]
        )
        blocks.append(
            f"Source: {record['source_question_id']} | score={record['source_score']} | outcome={record['outcome']}\n{items}"
        )
    return (
        "Consolidate the following Trace2Skill memory items into a compact SKILL.md for a "
        "multimodal visual/tool-using agent.\n\n"
        "Use this structure:\n"
        "---\n"
        "name: trace2skill-adapted\n"
        "description: Use when solving multimodal benchmark tasks with web, image, visit, and code tools\n"
        "---\n\n"
        "# Trace2Skill Adapted Skill\n"
        "## Visual Grounding\n"
        "## Tool Use\n"
        "## Verification\n"
        "## Failure Avoidance\n\n"
        "Keep it compact and actionable. Do not mention source ids, ground-truth labels, or sample-specific facts.\n\n"
        + "\n\n".join(blocks)
    )


def _build_trace2skill_guide(config, args) -> str:
    guide_path = _guide_path(args)
    if guide_path.exists():
        if getattr(args, "reuse_train_benchmark", ""):
            print(f"[Trace2Skill] Reusing skill from {args.reuse_train_benchmark}: {guide_path}")
        return guide_path.read_text(encoding="utf-8")
    if getattr(args, "reuse_train_benchmark", ""):
        raise FileNotFoundError(
            f"Missing reusable Trace2Skill skill for {args.benchmark}: {guide_path}. "
            f"Run {args.reuse_train_benchmark} before {args.benchmark}."
        )

    lesson_records = _collect_local_lessons(config, args)
    if not lesson_records:
        guide_path.parent.mkdir(parents=True, exist_ok=True)
        guide_path.write_text("", encoding="utf-8")
        return ""

    guide = _chat_text(
        args.model_name,
        TRACE2SKILL_CONSOLIDATION_SYSTEM_PROMPT,
        _consolidation_prompt(lesson_records),
        max_tokens=int(args.trace2skill_guide_max_tokens),
    )
    guide_path.parent.mkdir(parents=True, exist_ok=True)
    guide_path.write_text(guide, encoding="utf-8")
    return guide


def _format_trace2skill_guide(guide: str) -> str:
    if not guide.strip():
        return ""
    return (
        "## Trace2Skill Adapted Skill\n"
        f"{guide.strip()}\n\n"
        "Use this guide as reusable strategy only. Always ground the final answer in the current image, "
        "question, and tool evidence.\n"
    )


def _run_trace2skill_test_sample(
    sample_idx: int,
    sample: dict[str, Any],
    total_samples: int,
    args,
    sampling_params: dict[str, Any],
    base_prompt: str,
    guide_prompt: str,
) -> tuple[int, list[dict[str, Any]]]:
    question_id = _question_id(sample, sample_idx)
    sample_dir = Path(args.output_dir) / question_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_results, pending = _load_completed_rollouts(sample_dir, args.rollouts_per_sample)
    if not pending:
        print(f"[Trace2Skill] Sample {sample_idx + 1}/{total_samples} {question_id}: already completed.")
        compute_and_save_sample_summary(question_id, sample_results, args.rollouts_per_sample, sample_dir)
        return sample_idx, sample_results

    args._sample_system_prompt = guide_prompt + base_prompt
    sample_results_by_idx = {int(r["_rollout_idx"]): r for r in sample_results}
    for rollout_idx in pending:
        print(f"[Trace2Skill] Sample {sample_idx + 1}/{total_samples} {question_id}: rollout {rollout_idx}")
        result = _run_rollout(sample, args, sampling_params, rollout_idx)
        if result and not result_has_transient_failure(result):
            sample_results_by_idx[rollout_idx] = result

    ordered_results = [sample_results_by_idx[idx] for idx in sorted(sample_results_by_idx)]
    compute_and_save_sample_summary(question_id, ordered_results, args.rollouts_per_sample, sample_dir)
    return sample_idx, ordered_results


def run_trace2skill_benchmark(config, args) -> list[dict[str, Any]]:
    """Create an adapted SKILL.md from train trajectories, then inject it for test."""
    base_prompt = config.base_system_prompt
    guide_prompt = _format_trace2skill_guide(_build_trace2skill_guide(config, args))
    total_samples = len(config.data)
    max_workers = max(1, min(int(args.num_workers), total_samples))
    print(f"[Trace2Skill] Testing {total_samples} samples with {max_workers} worker(s).")

    sample_outputs: dict[int, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_trace2skill_test_sample,
                sample_idx,
                sample,
                total_samples,
                SimpleNamespace(**vars(args)),
                config.sampling_params,
                base_prompt,
                guide_prompt,
            )
            for sample_idx, sample in enumerate(config.data)
        ]
        for future in as_completed(futures):
            sample_idx, sample_results = future.result()
            sample_outputs[sample_idx] = sample_results

    all_results: list[dict[str, Any]] = []
    for sample_idx in sorted(sample_outputs):
        all_results.extend(sample_outputs[sample_idx])
    return all_results
