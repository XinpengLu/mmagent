"""ExpeL-style rule-memory baseline built on the existing rollout runner."""

from __future__ import annotations

import json
import math
import re
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
    _run_rollout,
    _sample_ground_truth,
    _sample_question,
)


EXPEL_SYSTEM_PROMPT = (
    "You extract reusable rules for visual reasoning agents. Given a task, ground truth, final answer, "
    "and judge analysis, produce short general rules that could help future similar tasks. "
    "Avoid sample-specific facts, names, and answers."
)


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


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9_]+", text.lower()) if len(token) > 2}


def _rule_similarity(question: str, rule: dict[str, Any]) -> float:
    query_tokens = _tokenize(question)
    rule_tokens = _tokenize(" ".join([str(rule.get("task", "")), str(rule.get("rule", ""))]))
    if not query_tokens or not rule_tokens:
        return 0.0
    return len(query_tokens & rule_tokens) / math.sqrt(len(query_tokens) * len(rule_tokens))


def _extract_rules_from_result(sample: dict[str, Any], result: dict[str, Any], args) -> list[str]:
    if result_has_transient_failure(result):
        return []
    prompt = (
        f"Question:\n{_sample_question(sample)}\n\n"
        f"Ground truth:\n{_sample_ground_truth(sample)}\n\n"
        f"Final answer:\n{result.get('final_answer', '')}\n\n"
        f"Score:\n{result.get('accuracy_score', 0.0)}\n\n"
        f"Judge analysis:\n{result.get('trajectory_analysis', '')}\n\n"
        "Return 1-3 reusable rules as plain bullet points."
    )
    text = _chat_text(args.model_name, EXPEL_SYSTEM_PROMPT, prompt, max_tokens=int(args.expel_rule_max_tokens))
    rules = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-*0123456789. ").strip()
        if cleaned:
            rules.append(cleaned)
    return rules[:3]


def _expel_rule_path(args) -> Path:
    return Path(args.output_dir).parent / "expel_rules.json"


def _build_expel_rules(config, args) -> list[dict[str, Any]]:
    rule_path = _expel_rule_path(args)
    if rule_path.exists():
        return list(read_json(rule_path).get("rules", []))

    train_file = Path(args.input_file).parent / "train.json"
    if not train_file.exists():
        print(f"[ExpeL] No train split for {train_file.parent.name}; using empty rule memory.")
        write_json(rule_path, {"rules": []}, indent=2)
        return []
    train_data = json.loads(train_file.read_text(encoding="utf-8"))[: int(args.expel_train_samples)]

    train_args = SimpleNamespace(**vars(args))
    train_args.input_file = str(train_file)
    train_args.output_dir = str(Path(args.output_dir).parent / "train_rule_extraction")
    train_args.rollouts_per_sample = 1
    train_args.num_workers = 1
    train_args._sample_system_prompt = config.base_system_prompt

    rules: list[dict[str, Any]] = []
    for sample_idx, sample in enumerate(train_data):
        question_id = _question_id(sample, sample_idx)
        print(f"[ExpeL] Training sample {sample_idx + 1}/{len(train_data)} {question_id}: collecting rule")
        result = _run_rollout(sample, train_args, config.sampling_params, rollout_idx=0)
        if not result:
            continue
        for rule_text in _extract_rules_from_result(sample, result, args):
            rules.append(
                {
                    "id": f"rule_{len(rules):04d}",
                    "task": _sample_question(sample),
                    "rule": rule_text,
                    "source_question_id": question_id,
                    "source_score": result.get("accuracy_score", 0.0),
                }
            )
    write_json(rule_path, {"rules": rules}, indent=2)
    return rules


def _selected_expel_rules(sample: dict[str, Any], rules: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    scored = [(_rule_similarity(_sample_question(sample), rule), rule) for rule in rules]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [rule for score, rule in scored[:top_k] if score > 0.0]


def _format_expel_rules(rules: list[dict[str, Any]]) -> str:
    if not rules:
        return ""
    text = "\n".join(f"{idx + 1}. {rule['rule']}" for idx, rule in enumerate(rules))
    return f"## Retrieved ExpeL Rules\n{text}\n\nUse these reusable rules as guidance, but rely on the current image and question.\n"


def run_expel_benchmark(config, args) -> list[dict[str, Any]]:
    """Train a lightweight rule memory, then retrieve rules for test rollouts."""
    base_prompt = config.base_system_prompt
    rules = _build_expel_rules(config, args)
    all_results: list[dict[str, Any]] = []

    for sample_idx, sample in enumerate(config.data):
        question_id = _question_id(sample, sample_idx)
        sample_dir = Path(args.output_dir) / question_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_results, pending = _load_completed_rollouts(sample_dir, args.rollouts_per_sample)
        if not pending:
            print(f"[ExpeL] Sample {sample_idx + 1}/{len(config.data)} {question_id}: already completed.")
            compute_and_save_sample_summary(question_id, sample_results, args.rollouts_per_sample, sample_dir)
            all_results.extend(sample_results)
            continue

        selected_rules = _selected_expel_rules(sample, rules, int(args.expel_top_k))
        args._sample_system_prompt = _format_expel_rules(selected_rules) + base_prompt
        write_json(sample_dir / "expel_selected_rules.json", {"rules": selected_rules}, indent=2)

        sample_results_by_idx = {int(r["_rollout_idx"]): r for r in sample_results}
        for rollout_idx in pending:
            print(f"[ExpeL] Sample {sample_idx + 1}/{len(config.data)} {question_id}: rollout {rollout_idx}")
            result = _run_rollout(sample, args, config.sampling_params, rollout_idx)
            if result and not result_has_transient_failure(result):
                sample_results_by_idx[rollout_idx] = result

        ordered_results = [sample_results_by_idx[idx] for idx in sorted(sample_results_by_idx)]
        compute_and_save_sample_summary(question_id, ordered_results, args.rollouts_per_sample, sample_dir)
        all_results.extend(ordered_results)

    return all_results
