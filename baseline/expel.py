"""ExpeL-style rule-memory baseline built on the existing rollout runner."""

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


EXPEL_SYSTEM_PROMPT = (
    "You are an advanced reasoning agent that updates a reusable rule set from past "
    "multimodal agent trajectories. Add, edit, remove, or agree with rules so the rule "
    "set stays general, concise, and useful for future tasks."
)

EXPEL_OPERATION_FORMAT = """Follow this exact operation format:

AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>
REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>
EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
ADD <NEW RULE NUMBER>: <NEW RULE>

Use at most 4 operations. Do not mention sample-specific facts, names, IDs, or answers.
Each rule must be a general strategy for future visual/tool-using reasoning tasks."""

EXPEL_RULE_MEMORY_VERSION = 3


def _empty_rule_memory() -> dict[str, Any]:
    return {"version": EXPEL_RULE_MEMORY_VERSION, "rules": [], "logs": []}


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


def _format_existing_rules(rules: list[dict[str, Any]]) -> str:
    if not rules:
        return "No existing rules yet."
    return "\n".join(f"{idx + 1}. {rule['rule']}" for idx, rule in enumerate(rules))


def _rule_operation_prompt(
    sample: dict[str, Any],
    result: dict[str, Any],
    rollout_dir: Path,
    rules: list[dict[str, Any]],
) -> str:
    score = float(result.get("accuracy_score", 0.0) or 0.0)
    outcome = "successful" if score >= 1.0 else "failed"
    return (
        f"Existing rules:\n{_format_existing_rules(rules)}\n\n"
        f"Trajectory outcome: {outcome}\n"
        f"Score: {score}\n\n"
        f"Question:\n{_sample_question(sample)}\n\n"
        f"Ground truth:\n{_sample_ground_truth(sample)}\n\n"
        f"Final answer:\n{result.get('final_answer', '')}\n\n"
        f"Judge analysis:\n{result.get('trajectory_analysis', '')}\n\n"
        f"Trajectory excerpt:\n{_trajectory_text(rollout_dir)}\n\n"
        "Update the existing rules based on this trajectory. "
        "For success, preserve or add strategies that caused success. "
        "For failure, add or edit rules that would prevent similar failures.\n\n"
        f"{EXPEL_OPERATION_FORMAT}"
    )


def _parse_rule_operations(text: str) -> list[tuple[str, int | None, str]]:
    operations = []
    for line in text.splitlines():
        match = re.match(r"^\s*(AGREE|REMOVE|EDIT|ADD)(?:\s+(\d+))?\s*:\s*(.+?)\s*$", line.strip(), re.I)
        if not match:
            continue
        op, index, rule_text = match.groups()
        rule_text = rule_text.strip()
        if rule_text:
            operations.append((op.upper(), int(index) if index else None, rule_text))
    return operations


def _rule_exists(rules: list[dict[str, Any]], rule_text: str) -> bool:
    return any(rule["rule"].lower() == rule_text.lower() for rule in rules)


def _renumber_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rules.sort(key=lambda rule: int(rule.get("count", 1)), reverse=True)
    for idx, rule in enumerate(rules):
        rule["id"] = f"rule_{idx:04d}"
    return rules


def _limit_rules(rules: list[dict[str, Any]], max_rules: int) -> list[dict[str, Any]]:
    ordered = sorted(rules, key=lambda rule: int(rule.get("count", 1)), reverse=True)[:max_rules]
    for idx, rule in enumerate(ordered):
        rule["id"] = f"rule_{idx:04d}"
    return ordered


def _apply_rule_operations(
    rules: list[dict[str, Any]],
    operations: list[tuple[str, int | None, str]],
    sample: dict[str, Any],
    question_id: str,
    score: float,
) -> list[dict[str, Any]]:
    for op, index, rule_text in operations:
        existing_idx = index - 1 if index is not None else None
        existing = rules[existing_idx] if existing_idx is not None and 0 <= existing_idx < len(rules) else None

        if op == "ADD":
            if not _rule_exists(rules, rule_text):
                rules.append(
                    {
                        "id": "",
                        "rule": rule_text,
                        "count": 2,
                        "source_question": _sample_question(sample),
                        "source_question_id": question_id,
                        "source_score": score,
                    }
                )
        elif op == "AGREE" and existing:
            existing["count"] = int(existing.get("count", 1)) + 1
        elif op == "EDIT" and existing:
            existing["rule"] = rule_text
            existing["count"] = int(existing.get("count", 1)) + 1
            existing["source_question"] = _sample_question(sample)
            existing["source_question_id"] = question_id
            existing["source_score"] = score
        elif op == "REMOVE" and existing:
            existing["count"] = int(existing.get("count", 1)) - 1

    return _renumber_rules([rule for rule in rules if int(rule.get("count", 1)) > 0])


def _update_rules_from_result(
    sample: dict[str, Any],
    result: dict[str, Any],
    args,
    rollout_dir: Path,
    memory: dict[str, Any],
    question_id: str,
) -> None:
    if result_has_transient_failure(result):
        return
    text = _chat_text(
        args.model_name,
        EXPEL_SYSTEM_PROMPT,
        _rule_operation_prompt(sample, result, rollout_dir, memory["rules"]),
        max_tokens=int(args.expel_rule_max_tokens),
    )
    operations = _parse_rule_operations(text)
    memory["rules"] = _limit_rules(
        _apply_rule_operations(
            memory["rules"],
            operations,
            sample,
            question_id,
            float(result.get("accuracy_score", 0.0) or 0.0),
        ),
        int(args.expel_max_rules),
    )
    memory["logs"].append({"source_question_id": question_id, "operations": text, "parsed_operations": operations})


def _expel_rule_path(args) -> Path:
    return Path(args.train_output_dir) / "expel_rule_memory.json"


def _run_expel_training_sample(
    sample_idx: int,
    sample: dict[str, Any],
    total_samples: int,
    train_args,
    config,
    memory: dict[str, Any],
    rule_path: Path,
    update_args,
) -> None:
    question_id = _question_id(sample, sample_idx)
    for rollout_idx in range(train_args.rollouts_per_sample):
        print(
            f"[ExpeL] Training sample {sample_idx + 1}/{total_samples} {question_id}: "
            f"trial {rollout_idx + 1}/{train_args.rollouts_per_sample}"
        )
        train_args._sample_system_prompt = _format_expel_rules(memory["rules"]) + config.base_system_prompt
        result = _run_rollout(sample, train_args, config.sampling_params, rollout_idx=rollout_idx)
        if not result or result_has_transient_failure(result):
            continue

        rollout_dir = _rollout_dir(train_args.output_dir, question_id, rollout_idx, train_args.rollouts_per_sample)
        _update_rules_from_result(sample, result, update_args, rollout_dir, memory, question_id)
        write_json(rule_path, memory, indent=2)

        if float(result.get("accuracy_score", 0.0) or 0.0) >= 1.0:
            break


def _build_expel_rules(config, args) -> list[dict[str, Any]]:
    train_file = Path(args.input_file).parent / "train.json"
    rule_path = _expel_rule_path(args)
    reuse_source = getattr(args, "reuse_train_benchmark", "")
    if rule_path.exists():
        memory = read_json(rule_path)
        if int(memory.get("version", 0)) == EXPEL_RULE_MEMORY_VERSION:
            if reuse_source:
                print(f"[ExpeL] Reusing rules from {reuse_source}: {rule_path}")
            return list(memory.get("rules", []))
    if reuse_source:
        raise FileNotFoundError(
            f"Missing reusable ExpeL rule memory for {args.benchmark}: {rule_path}. "
            f"Run {reuse_source} before {args.benchmark}."
        )

    train_data = json.loads(train_file.read_text(encoding="utf-8"))[: int(args.expel_train_samples)]

    train_args = SimpleNamespace(**vars(args))
    train_args.phase = "train"
    train_args.input_file = str(train_file)
    train_args.output_dir = str(Path(args.train_output_dir))
    train_args.rollouts_per_sample = int(args.rollouts_per_sample)
    train_args.num_workers = 1
    train_args._sample_system_prompt = config.base_system_prompt

    memory = _empty_rule_memory()
    for sample_idx, sample in enumerate(train_data):
        _run_expel_training_sample(
            sample_idx,
            sample,
            len(train_data),
            train_args,
            config,
            memory,
            rule_path,
            args,
        )
    return memory["rules"]


def _format_expel_rules(rules: list[dict[str, Any]]) -> str:
    if not rules:
        return ""
    text = "\n".join(f"{idx + 1}. {rule['rule']}" for idx, rule in enumerate(rules))
    return f"## Retrieved ExpeL Rules\n{text}\n\nUse these reusable rules as guidance, but rely on the current image and question.\n"


def _run_expel_test_sample(
    sample_idx: int,
    sample: dict[str, Any],
    total_samples: int,
    args,
    sampling_params: dict[str, Any],
    base_prompt: str,
    rules: list[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    question_id = _question_id(sample, sample_idx)
    sample_dir = Path(args.output_dir) / question_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_results, pending = _load_completed_rollouts(sample_dir, args.rollouts_per_sample)
    if not pending:
        print(f"[ExpeL] Sample {sample_idx + 1}/{total_samples} {question_id}: already completed.")
        compute_and_save_sample_summary(question_id, sample_results, args.rollouts_per_sample, sample_dir)
        return sample_idx, sample_results

    args._sample_system_prompt = _format_expel_rules(rules) + base_prompt
    write_json(sample_dir / "expel_injected_rules.json", {"rules": rules}, indent=2)

    sample_results_by_idx = {int(r["_rollout_idx"]): r for r in sample_results}
    for rollout_idx in pending:
        print(f"[ExpeL] Sample {sample_idx + 1}/{total_samples} {question_id}: rollout {rollout_idx}")
        result = _run_rollout(sample, args, sampling_params, rollout_idx)
        if result and not result_has_transient_failure(result):
            sample_results_by_idx[rollout_idx] = result

    ordered_results = [sample_results_by_idx[idx] for idx in sorted(sample_results_by_idx)]
    compute_and_save_sample_summary(question_id, ordered_results, args.rollouts_per_sample, sample_dir)
    return sample_idx, ordered_results


def run_expel_benchmark(config, args) -> list[dict[str, Any]]:
    """Train an ExpeL-style rule memory, then inject the highest-count rules for test."""
    base_prompt = config.base_system_prompt
    rules = _build_expel_rules(config, args)
    total_samples = len(config.data)
    max_workers = max(1, min(int(args.num_workers), total_samples))
    print(f"[ExpeL] Testing {total_samples} samples with {max_workers} worker(s).")

    sample_outputs: dict[int, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_expel_test_sample,
                sample_idx,
                sample,
                total_samples,
                SimpleNamespace(**vars(args)),
                config.sampling_params,
                base_prompt,
                rules,
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
