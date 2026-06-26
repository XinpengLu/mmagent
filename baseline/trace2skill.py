"""Trace2Skill-lite baseline built on the existing rollout runner."""

from __future__ import annotations

import json
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


TRACE2SKILL_LESSON_SYSTEM_PROMPT = (
    "You are a Trace2Skill analyst for multimodal tool-using agents. "
    "Distill one execution trajectory into reusable, task-agnostic lessons. "
    "Focus on visual evidence use, web/tool strategy, verification, and common mistakes. "
    "Do not include sample-specific answers or facts."
)

TRACE2SKILL_CONSOLIDATION_SYSTEM_PROMPT = (
    "You are consolidating trajectory-local lessons into a compact skill guide for a "
    "multimodal tool-using agent. Merge duplicates, resolve conflicts, and keep only "
    "general strategies that transfer across tasks."
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
    return _trace2skill_root(args) / "trace2skill_local_lessons.json"


def _guide_path(args) -> Path:
    return _trace2skill_root(args) / "trace2skill_guide.md"


def _lesson_prompt(sample: dict[str, Any], result: dict[str, Any], rollout_dir: Path) -> str:
    outcome = "success" if float(result.get("accuracy_score", 0.0) or 0.0) >= 1.0 else "failure"
    return (
        f"Task outcome: {outcome}\n\n"
        f"Question:\n{_sample_question(sample)}\n\n"
        f"Ground truth:\n{_sample_ground_truth(sample)}\n\n"
        f"Final answer:\n{result.get('final_answer', '')}\n\n"
        f"Judge analysis:\n{result.get('trajectory_analysis', '')}\n\n"
        f"Trajectory excerpt:\n{_trajectory_text(rollout_dir)}\n\n"
        "Return 2-4 concise bullet lessons. Each lesson should be generally useful for future tasks."
    )


def _extract_local_lessons(sample: dict[str, Any], result: dict[str, Any], args, rollout_dir: Path) -> list[str]:
    text = _chat_text(
        args.model_name,
        TRACE2SKILL_LESSON_SYSTEM_PROMPT,
        _lesson_prompt(sample, result, rollout_dir),
        max_tokens=int(args.trace2skill_lesson_max_tokens),
    )
    lessons = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-*0123456789. ").strip()
        if cleaned:
            lessons.append(cleaned)
    return lessons[:4]


def _collect_local_lessons(config, args) -> list[dict[str, Any]]:
    train_file = Path(args.input_file).parent / "train.json"
    lesson_path = _local_lessons_path(args)
    if lesson_path.exists():
        return list(read_json(lesson_path).get("lessons", []))

    train_data = json.loads(train_file.read_text(encoding="utf-8"))[: int(args.trace2skill_train_samples)]
    train_args = SimpleNamespace(**vars(args))
    train_args.phase = "train"
    train_args.input_file = str(train_file)
    train_args.output_dir = str(_trace2skill_root(args))
    train_args.rollouts_per_sample = 1
    train_args.num_workers = 1
    train_args._sample_system_prompt = config.base_system_prompt

    lesson_records: list[dict[str, Any]] = []
    for sample_idx, sample in enumerate(train_data):
        question_id = _question_id(sample, sample_idx)
        print(f"[Trace2Skill] Training sample {sample_idx + 1}/{len(train_data)} {question_id}: extracting lessons")
        result = _run_rollout(sample, train_args, config.sampling_params, rollout_idx=0)
        if not result or result_has_transient_failure(result):
            continue
        rollout_dir = _rollout_dir(train_args.output_dir, question_id, 0, train_args.rollouts_per_sample)
        lessons = _extract_local_lessons(sample, result, args, rollout_dir)
        if lessons:
            lesson_records.append(
                {
                    "source_question_id": question_id,
                    "source_score": result.get("accuracy_score", 0.0),
                    "lessons": lessons,
                }
            )

    write_json(lesson_path, {"lessons": lesson_records}, indent=2)
    return lesson_records


def _consolidation_prompt(lesson_records: list[dict[str, Any]]) -> str:
    blocks = []
    for record in lesson_records:
        lessons = "\n".join(f"- {lesson}" for lesson in record["lessons"])
        blocks.append(
            f"Source: {record['source_question_id']} | score={record['source_score']}\n{lessons}"
        )
    return (
        "Consolidate the following trajectory-local lessons into a short markdown skill guide.\n\n"
        "Use this structure:\n"
        "# Trace2Skill Guide\n"
        "## Visual Grounding\n"
        "## Tool Use\n"
        "## Verification\n"
        "## Failure Avoidance\n\n"
        "Keep it compact and actionable. Do not mention source ids or sample-specific facts.\n\n"
        + "\n\n".join(blocks)
    )


def _build_trace2skill_guide(config, args) -> str:
    guide_path = _guide_path(args)
    if guide_path.exists():
        return guide_path.read_text(encoding="utf-8")

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
        "## Trace2Skill Global Skill Guide\n"
        f"{guide.strip()}\n\n"
        "Use this guide as reusable strategy only. Always ground the final answer in the current image, "
        "question, and tool evidence.\n"
    )


def run_trace2skill_benchmark(config, args) -> list[dict[str, Any]]:
    """Create a global skill guide from train trajectories, then inject it for test."""
    base_prompt = config.base_system_prompt
    guide_prompt = _format_trace2skill_guide(_build_trace2skill_guide(config, args))
    all_results: list[dict[str, Any]] = []

    for sample_idx, sample in enumerate(config.data):
        question_id = _question_id(sample, sample_idx)
        sample_dir = Path(args.output_dir) / question_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_results, pending = _load_completed_rollouts(sample_dir, args.rollouts_per_sample)
        if not pending:
            print(f"[Trace2Skill] Sample {sample_idx + 1}/{len(config.data)} {question_id}: already completed.")
            compute_and_save_sample_summary(question_id, sample_results, args.rollouts_per_sample, sample_dir)
            all_results.extend(sample_results)
            continue

        args._sample_system_prompt = guide_prompt + base_prompt
        sample_results_by_idx = {int(r["_rollout_idx"]): r for r in sample_results}
        for rollout_idx in pending:
            print(f"[Trace2Skill] Sample {sample_idx + 1}/{len(config.data)} {question_id}: rollout {rollout_idx}")
            result = _run_rollout(sample, args, config.sampling_params, rollout_idx)
            if result and not result_has_transient_failure(result):
                sample_results_by_idx[rollout_idx] = result

        ordered_results = [sample_results_by_idx[idx] for idx in sorted(sample_results_by_idx)]
        compute_and_save_sample_summary(question_id, ordered_results, args.rollouts_per_sample, sample_dir)
        all_results.extend(ordered_results)

    return all_results
