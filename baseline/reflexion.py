"""Reflexion baseline built on the existing rollout runner."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from engine.api_caller import call_vision_api
from main_flow.metrics import compute_and_save_sample_summary
from utils.artifacts import (
    ROLLOUT_METRICS_FILENAME,
    TRAJECTORY_FILENAME,
    clean_rollout_artifacts,
    read_json,
    write_json,
)
from utils.transient import result_has_transient_failure


REFLEXION_SYSTEM_PROMPT = (
    "You are analyzing a failed visual-agent attempt. Write one concise reflection that identifies "
    "the likely mistake and gives a general strategy for the next attempt. Do not solve the problem "
    "directly. Return only the reflection."
)


def _question_id(sample: dict[str, Any], sample_idx: int) -> str:
    return str(sample.get("doc_id") or sample.get("question_id") or f"sample_{sample_idx}")


def _sample_question(sample: dict[str, Any]) -> str:
    return str(sample.get("problem") or sample.get("question") or "")


def _sample_ground_truth(sample: dict[str, Any]) -> str:
    return str(sample.get("solution") or sample.get("answer") or sample.get("ground_truth") or "")


def _rollout_dir(output_dir: str | Path, question_id: str, rollout_idx: int, rollouts_per_sample: int) -> Path:
    sample_dir = Path(output_dir) / question_id
    return sample_dir / f"rollout_{rollout_idx}" if rollouts_per_sample > 1 else sample_dir


def _completed_rollout_result(sample_dir: Path, rollout_idx: int, rollouts_per_sample: int) -> dict[str, Any] | None:
    rollout_dir = sample_dir / f"rollout_{rollout_idx}" if rollouts_per_sample > 1 else sample_dir
    traj_path = rollout_dir / TRAJECTORY_FILENAME
    metrics_path = rollout_dir / ROLLOUT_METRICS_FILENAME
    if not (traj_path.exists() and metrics_path.exists()):
        return None
    try:
        metrics = read_json(metrics_path)
    except (OSError, json.JSONDecodeError, ValueError):
        clean_rollout_artifacts(rollout_dir)
        return None
    if not isinstance(metrics, dict):
        clean_rollout_artifacts(rollout_dir)
        return None
    if metrics.get("api_transient_failure") or metrics.get("tool_transient_failure"):
        return None
    return {
        "question_id": sample_dir.name,
        "accuracy_score": float(metrics.get("accuracy_score", 0.0) or 0.0),
        "conversation_history": [{"role": "assistant", "content": ""}],
        "_rollout_idx": rollout_idx,
    }


def _load_completed_rollouts(sample_dir: Path, rollouts_per_sample: int) -> tuple[list[dict[str, Any]], list[int]]:
    results = []
    missing = []
    for rollout_idx in range(rollouts_per_sample):
        result = _completed_rollout_result(sample_dir, rollout_idx, rollouts_per_sample)
        if result is None:
            missing.append(rollout_idx)
        else:
            results.append(result)
    return results, missing


def _trajectory_text(rollout_dir: Path, max_chars: int = 6000) -> str:
    traj_path = rollout_dir / TRAJECTORY_FILENAME
    if not traj_path.exists():
        return ""
    lines = []
    with traj_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(line.strip())
    return "\n".join(lines)[-max_chars:]


def _chat_text(model_name: str, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
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


def _reflection_prompt(sample: dict[str, Any], result: dict[str, Any], rollout_dir: Path) -> str:
    return (
        f"Question:\n{_sample_question(sample)}\n\n"
        f"Ground truth:\n{_sample_ground_truth(sample)}\n\n"
        f"Final answer:\n{result.get('final_answer', '')}\n\n"
        f"Judge analysis:\n{result.get('trajectory_analysis', '')}\n\n"
        f"Trajectory excerpt:\n{_trajectory_text(rollout_dir)}"
    )


def _format_reflections(reflections: list[str]) -> str:
    if not reflections:
        return ""
    items = "\n".join(f"{idx + 1}. {text}" for idx, text in enumerate(reflections))
    return f"## Reflections From Previous Attempts\n{items}\n\nUse these reflections to avoid repeating earlier mistakes.\n"


def _run_rollout(sample, args, sampling_params, rollout_idx: int) -> dict[str, Any] | None:
    from main_flow.rollout_runner import _run_rollout_until_valid

    result = _run_rollout_until_valid(sample, args, sampling_params, rollout_idx=rollout_idx)
    if result:
        result["_rollout_idx"] = rollout_idx
    return result


def _run_reflexion_sample(
    sample_idx: int,
    sample: dict[str, Any],
    total_samples: int,
    args,
    sampling_params: dict[str, Any],
    base_prompt: str,
) -> tuple[int, list[dict[str, Any]]]:
    question_id = _question_id(sample, sample_idx)
    sample_dir = Path(args.output_dir) / question_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_results, pending = _load_completed_rollouts(sample_dir, args.rollouts_per_sample)

    if not pending:
        print(f"[Reflexion] Sample {sample_idx + 1}/{total_samples} {question_id}: already completed.")
        compute_and_save_sample_summary(question_id, sample_results, args.rollouts_per_sample, sample_dir)
        return sample_idx, sample_results

    reflections: list[str] = []
    sample_results_by_idx = {int(r["_rollout_idx"]): r for r in sample_results}
    for rollout_idx in pending:
        print(f"[Reflexion] Sample {sample_idx + 1}/{total_samples} {question_id}: rollout {rollout_idx}")
        args._sample_system_prompt = _format_reflections(reflections) + base_prompt
        result = _run_rollout(sample, args, sampling_params, rollout_idx)
        if not result or result_has_transient_failure(result):
            continue

        sample_results_by_idx[rollout_idx] = result
        if float(result.get("accuracy_score", 0.0) or 0.0) < 1.0 and rollout_idx < args.rollouts_per_sample - 1:
            reflection = _chat_text(
                args.model_name,
                REFLEXION_SYSTEM_PROMPT,
                _reflection_prompt(sample, result, _rollout_dir(args.output_dir, question_id, rollout_idx, args.rollouts_per_sample)),
                max_tokens=int(args.reflection_max_tokens),
            )
            if reflection:
                reflections.append(reflection)
                write_json(sample_dir / "reflexion_memory.json", {"reflections": reflections}, indent=2)

    ordered_results = [sample_results_by_idx[idx] for idx in sorted(sample_results_by_idx)]
    compute_and_save_sample_summary(question_id, ordered_results, args.rollouts_per_sample, sample_dir)
    if len(ordered_results) < args.rollouts_per_sample:
        return sample_idx, []
    return sample_idx, ordered_results


def run_reflexion_benchmark(config, args) -> list[dict[str, Any]]:
    """Run Reflexion with parallel samples and sequential rollouts per sample."""
    total_samples = len(config.data)
    max_workers = max(1, min(int(args.num_workers), total_samples))
    sample_outputs: dict[int, list[dict[str, Any]]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_reflexion_sample,
                sample_idx,
                sample,
                total_samples,
                SimpleNamespace(**vars(args)),
                config.sampling_params,
                config.base_system_prompt,
            )
            for sample_idx, sample in enumerate(config.data)
        ]
        for future in as_completed(futures):
            sample_idx, results = future.result()
            sample_outputs[sample_idx] = results

    all_results: list[dict[str, Any]] = []
    for sample_idx in sorted(sample_outputs):
        all_results.extend(sample_outputs[sample_idx])
    return all_results
