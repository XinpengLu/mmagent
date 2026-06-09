"""Train-time experience generation pipeline."""

import json
import os
import re

from .critique import intra_sample_experiences, supervised_correction_experiences
from .llm_client import ExperienceLLM
from .schema import _compact_experience_operation, _normalize_experience_operation
from .trajectory_summary import summarize_rollouts
from .utils import append_sample_metadata_tags, sample_metadata_tags
from utils.artifacts import (
    RETRIEVAL_INFO_FILENAME,
    TRAJECTORY_FILENAME,
    iter_jsonl,
)


def _operation_kind(operation):
    """Return the explicit kind on an LLM-produced experience operation, if any."""
    if not isinstance(operation, dict):
        return None
    payload = operation.get("experience")
    if isinstance(payload, dict):
        value = payload.get("kind") or payload.get("outcome")
    else:
        value = operation.get("kind") or operation.get("outcome")
    return str(value).strip().lower() if value is not None else None


def _operation_record(operation):
    """Return the experience payload regardless of add/modify wrapper shape."""
    if not isinstance(operation, dict):
        return {}
    payload = operation.get("experience")
    return payload if isinstance(payload, dict) else operation


def _has_executable_success_record(record):
    """Check whether a success memory contains enough executable evidence."""
    if not isinstance(record, dict) or str(record.get("kind", "")).lower() != "success":
        return False
    code = str(record.get("code_template") or "")
    if len(code) >= 160:
        return True
    sequence = record.get("tool_sequence") or []
    if sequence and any(isinstance(step, dict) and step.get("purpose") for step in sequence):
        content = str(record.get("content") or "").lower()
        return not any(
            vague in content
            for vague in (
                "similar visual reasoning problem",
                "verified tool-backed solution",
                "reuse successful",
                "tool-derived evidence",
            )
        )
    return False


def _parse_retrieval_exp_blocks(section):
    """Parse [id] blocks from the human-readable retrieval artifact."""
    result = {}
    exp_blocks = re.split(r"\n\[([^\]]+)\]\n", section)
    for i in range(1, len(exp_blocks), 2):
        exp_id = exp_blocks[i]
        exp_content = exp_blocks[i + 1].strip().strip("-").strip() if i + 1 < len(exp_blocks) else ""
        if exp_content:
            result[exp_id] = exp_content
    return result


def _parse_retrieval_exp_ids(section):
    return set(re.findall(r"\[([^\]]+)\]", section))


def _get_used_original_experiences(sample_dir):
    """Return original retrieved experiences that survived optional rewrite."""
    retrieval_info_file = os.path.join(sample_dir, RETRIEVAL_INFO_FILENAME)
    if not os.path.exists(retrieval_info_file):
        return {}

    try:
        with open(retrieval_info_file, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return {}

    original_exps = {}
    rewritten_ids = set()
    if "ORIGINAL RETRIEVED EXPERIENCES" in content:
        original_section = content.split("ORIGINAL RETRIEVED EXPERIENCES", 1)[1]
        if "REWRITTEN EXPERIENCES" in original_section:
            original_section = original_section.split("REWRITTEN EXPERIENCES", 1)[0]
        elif "RETRIEVED EXPERIENCE CONTENT" in original_section:
            original_section = original_section.split("RETRIEVED EXPERIENCE CONTENT", 1)[0]
        original_exps = _parse_retrieval_exp_blocks(original_section)

    if "REWRITTEN EXPERIENCES" in content:
        rewritten_section = content.split("REWRITTEN EXPERIENCES", 1)[1]
        rewritten_ids = _parse_retrieval_exp_ids(rewritten_section)
    elif "RETRIEVED EXPERIENCE CONTENT" in content:
        retrieved_section = content.split("RETRIEVED EXPERIENCE CONTENT", 1)[1]
        rewritten_ids = _parse_retrieval_exp_ids(retrieved_section)
        if not original_exps:
            original_exps = _parse_retrieval_exp_blocks(retrieved_section)
            rewritten_ids = set(original_exps.keys())

    return {exp_id: text for exp_id, text in original_exps.items() if exp_id in rewritten_ids}


def _short_code_template(code, max_lines=80, max_chars=6000):
    """Keep the executable core from a successful code tool call."""
    lines = []
    for line in str(code or "").splitlines():
        stripped = line.rstrip()
        if not stripped:
            continue
        lines.append(stripped)
        if len(lines) >= max_lines or len("\n".join(lines)) >= max_chars:
            break
    return "\n".join(lines)[:max_chars].rstrip()


def _score_success_code(code, result_text="", call_index=0, total_calls=1):
    """Prefer code that computes or validates an answer over display-only probes."""
    text = str(code or "")
    lowered = text.lower()
    if not lowered.strip():
        return -999.0

    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    score = min(len(non_empty_lines), 80) * 0.04
    score += min(len(text), 5000) / 2500.0

    total_calls = max(int(total_calls or 1), 1)
    score += 0.6 * (call_index / max(total_calls - 1, 1))

    result_lower = str(result_text or "").lower()
    if "traceback" in result_lower or "syntaxerror" in result_lower or "nameerror" in result_lower:
        score -= 4.0
    return score


def _select_best_success_code(records):
    """Return the most reusable successful code call from one trajectory."""
    code_calls = []
    for idx, record in enumerate(records):
        tool_call = record.get("tool_call") if isinstance(record, dict) else None
        if not isinstance(tool_call, dict):
            continue
        tool_name = str(tool_call.get("tool_name") or tool_call.get("tool") or "").strip()
        if tool_name != "code_interpreter":
            continue
        params = tool_call.get("parameters") if isinstance(tool_call.get("parameters"), dict) else {}
        code = params.get("code")
        if not code:
            continue
        code_calls.append(
            {
                "index": idx,
                "code": str(code),
                "result": str(tool_call.get("result") or ""),
            }
        )
    if not code_calls:
        return None
    total = len(code_calls)
    for local_idx, call in enumerate(code_calls):
        call["score"] = _score_success_code(
            call["code"],
            call.get("result", ""),
            call_index=local_idx,
            total_calls=total,
        )
    return max(code_calls, key=lambda item: item["score"])


def _best_success_rollout_code(successful_rollouts):
    best_rollout = None
    best_code_call = None
    best_records = []

    for rollout in successful_rollouts:
        records = list(iter_jsonl(rollout.get("traj_path")))
        code_call = _select_best_success_code(records)
        if not code_call:
            continue
        score = code_call.get("score", 0.0) + float(rollout.get("accuracy_score", 0.0) or 0.0)
        if best_code_call is None or score > best_code_call.get("score", -999.0):
            best_rollout = rollout
            best_code_call = dict(code_call)
            best_code_call["score"] = score
            best_records = records

    return best_rollout, best_code_call, best_records


def _initial_success_tool_plan(best_code_call):
    tool_sequence = []
    tool_tags = []
    code_template = ""
    operation = ""

    if not best_code_call:
        return tool_sequence, tool_tags, code_template, operation

    code_template = _short_code_template(best_code_call.get("code"))
    operation = "run the successful tool-backed analysis and validate its intermediate evidence"
    tool_tags.append("code_interpreter")
    tool_sequence.append(
        {
            "tool": "code_interpreter",
            "purpose": operation,
            "expected_output": "a visible or printed intermediate result that directly constrains the final answer",
        }
    )
    return tool_sequence, tool_tags, code_template, operation


def _extend_success_tool_plan(source_records, tool_sequence, tool_tags, operation):
    for record in source_records:
        tool_call = record.get("tool_call") if isinstance(record, dict) else None
        if not isinstance(tool_call, dict):
            continue
        tool_name = str(tool_call.get("tool_name") or tool_call.get("tool") or "").strip()
        if not tool_name:
            continue
        if tool_name not in tool_tags:
            tool_tags.append(tool_name)
        if tool_name == "code_interpreter":
            purpose = operation or "run the successful image-analysis code and inspect the result"
            expected = "tool output that supports or rejects the candidate answer"
        else:
            purpose = "reuse the successful tool-backed evidence gathering step from the correct rollout"
            expected = "evidence that can be checked before answering"
        step = {"tool": tool_name, "purpose": purpose, "expected_output": expected}
        if step not in tool_sequence:
            tool_sequence.append(step)
        if len(tool_sequence) >= 5:
            break


def _auto_success_record(sample_info, tool_tags, tool_sequence, operation, source_rollouts, code_template):
    if not operation:
        operation = "collect and validate tool-backed visual evidence"

    task = sample_info.get("sample", {}).get("task")
    task_tags = [str(task)] if task else ["visual_reasoning"]
    experience = {
        "kind": "success",
        "task_tags": task_tags,
        "tool_tags": tool_tags,
        "content": (
            f"When the question requires {operation}, use code_interpreter to reproduce that "
            "operation on the current image and inspect the intermediate result before answering."
        ),
        "tool_sequence": tool_sequence,
        "validation_checks": [
            "Confirm the tool output directly narrows the candidate answer.",
            "Reject a direct visual guess when it conflicts with the reproduced tool evidence.",
            "Check that any crop, threshold, grid, or measurement is derived from the current image, not copied blindly.",
        ],
        "fallback_plan": [
            "If the first tool output is empty or visually inconsistent, adjust the crop, threshold, or candidate set and rerun once.",
            "If two tool-derived candidates disagree, choose the one supported by the clearer visible intermediate evidence.",
        ],
        "evidence_rollouts": source_rollouts,
        "confidence": 0.78,
    }
    if code_template:
        experience["code_template"] = code_template
    return experience


def _auto_success_experience_from_rollouts(question_id, sample_info, evidence_rollouts):
    """
    If the LLM critique misses all success memories, synthesize a minimal one
    from actual successful trajectory tool calls. This preserves self-evolution
    without injecting task-specific expert rules.
    """
    successful = [r for r in evidence_rollouts if r.get("success") and r.get("traj_path")]
    if not successful:
        return None

    best_rollout, best_code_call, best_records = _best_success_rollout_code(successful)
    tool_sequence, tool_tags, code_template, operation = _initial_success_tool_plan(best_code_call)
    source_records = best_records if best_records else list(iter_jsonl(successful[0].get("traj_path")))
    _extend_success_tool_plan(source_records, tool_sequence, tool_tags, operation)

    if not tool_sequence:
        return None

    source_rollouts = [best_rollout] if best_rollout else successful[:1]
    experience = _auto_success_record(
        sample_info,
        tool_tags,
        tool_sequence,
        operation,
        source_rollouts,
        code_template,
    )
    return {"option": "add", "experience": experience}


def _initial_generation_result(question_id):
    return {
        'sample_id': question_id,
        'success': False,
        'experience_ops': [],
        'skill_content': '',
        'error': None,
    }


def _trajectory_paths(sample_dir, sample_rollout_results, args):
    paths = []
    for rollout_idx in range(len(sample_rollout_results)):
        rollout_dir = (
            os.path.join(sample_dir, f"rollout_{rollout_idx}")
            if args.rollouts_per_sample > 1
            else sample_dir
        )
        traj_path = os.path.join(rollout_dir, TRAJECTORY_FILENAME)
        if os.path.exists(traj_path):
            paths.append(traj_path)
    return paths


def _trajectory_has_turns(traj_paths):
    return any(
        rec.get("turn_idx") is not None
        for traj_path in traj_paths
        for rec in iter_jsonl(traj_path)
    )


def _summary_fields(merged_summaries, sample_rollout_results):
    question = merged_summaries.get('question', '') or sample_rollout_results[0].get('prompt', '')
    ground_truth = merged_summaries.get('ground_truth', '') or sample_rollout_results[0].get('ground_truth', '')
    system_prompt_text = merged_summaries.get('system_prompt', '')
    summaries_only = {
        k: v
        for k, v in merged_summaries.items()
        if k not in ['question', 'ground_truth', 'system_prompt']
    }
    return question, ground_truth, system_prompt_text, summaries_only


def _rollout_scores(sample_rollout_results):
    return [
        r.get("accuracy_score")
        for r in sample_rollout_results
        if isinstance(r, dict) and r.get("accuracy_score") is not None
    ]


def _evidence_rollouts(question_id, sample_dir, sample_rollout_results, args):
    evidence = []
    for fallback_idx, rollout_result in enumerate(sample_rollout_results):
        if not isinstance(rollout_result, dict):
            continue
        rollout_idx = rollout_result.get("_rollout_idx", fallback_idx)
        rollout_dir = (
            os.path.join(sample_dir, f"rollout_{rollout_idx}")
            if args.rollouts_per_sample > 1
            else sample_dir
        )
        score = rollout_result.get("accuracy_score")
        evidence.append({
            "doc_id": question_id,
            "rollout_id": f"{question_id}:rollout_{rollout_idx}",
            "rollout_idx": rollout_idx,
            "traj_path": os.path.join(rollout_dir, TRAJECTORY_FILENAME),
            "accuracy_score": score,
            "success": bool(score == 1.0),
            "final_answer": rollout_result.get("final_answer"),
        })
    return evidence


def _success_state(score_list):
    success_count = sum(1 for score in score_list if float(score) == 1.0)
    total_rollouts = max(1, len(score_list))
    has_success = success_count > 0
    reliable_success = success_count > (total_rollouts / 2)
    return {
        "has_success": has_success,
        "all_rollouts_failed": not has_success,
        "default_kind": "success" if reliable_success else "failure",
    }


def _experience_source(question_id, sample_info, score_list, evidence_rollouts):
    sample = sample_info.get("sample", {})
    return {
        "doc_id": question_id,
        "task": sample.get("task"),
        "metadata_tags": sample_metadata_tags(sample),
        "accuracy_scores": score_list,
        "evidence_rollouts": evidence_rollouts,
    }


def _iter_raw_experience_ops(raw_ops):
    if isinstance(raw_ops, str):
        try:
            raw_ops = json.loads(raw_ops)
        except Exception:
            return []

    if isinstance(raw_ops, dict) and 'experiences' in raw_ops:
        return raw_ops['experiences'].values()
    if isinstance(raw_ops, list):
        return raw_ops
    if isinstance(raw_ops, dict):
        return [raw_ops]
    return []


def _normalize_llm_experience_ops(raw_ops, question_id, sample_info, source, default_kind, has_success, all_rollouts_failed):
    norm_ops = []
    sample = sample_info.get("sample", {})
    for op in _iter_raw_experience_ops(raw_ops):
        if isinstance(op, str):
            op = {"experience": op}
        elif isinstance(op, dict):
            op["task_tags"] = append_sample_metadata_tags(op.get("task_tags") or [], sample)
            if _operation_kind(op) == "success" and not has_success:
                print(
                    f"  [Experience] Dropping success memory for {question_id} "
                    f"because no rollout was successful"
                )
                continue
            if all_rollouts_failed:
                op["kind"] = "failure"
                if op.get("confidence") is not None:
                    try:
                        op["confidence"] = min(float(op["confidence"]), 0.65)
                    except (TypeError, ValueError):
                        op["confidence"] = 0.6

        normalized = _normalize_experience_operation(op, default_kind=default_kind, source=source)
        if normalized:
            norm_ops.append(normalized)
    return norm_ops


def _add_auto_success_memory(norm_ops, question_id, sample_info, source, evidence_rollouts, has_success):
    has_executable_success_memory = any(
        _has_executable_success_record(_operation_record(op))
        for op in norm_ops
        if isinstance(op, dict)
    )
    if not has_success or has_executable_success_memory:
        return

    auto_success_op = _auto_success_experience_from_rollouts(question_id, sample_info, evidence_rollouts)
    if not auto_success_op:
        return

    auto_success_op["task_tags"] = append_sample_metadata_tags(
        auto_success_op.get("task_tags") or [],
        sample_info.get("sample", {}),
    )
    normalized = _normalize_experience_operation(auto_success_op, default_kind="success", source=source)
    if normalized:
        norm_ops.append(normalized)
        print(
            f"  [Experience] Added auto success memory for {question_id} "
            f"from successful rollout tool calls"
        )


def _add_supervised_correction_ops(
    norm_ops,
    question_id,
    question,
    ground_truth,
    summaries_only,
    llm,
    sample_info,
    sample_dir,
    system_prompt_text,
    used_experiences,
    source,
    evidence_rollouts,
    all_rollouts_failed,
):
    if not all_rollouts_failed or not str(ground_truth or "").strip():
        return

    correction_ops = supervised_correction_experiences(
        question,
        ground_truth,
        summaries_only,
        llm,
        debug_dir=sample_dir,
        system_prompt=system_prompt_text,
        used_experiences=used_experiences,
    )
    sample = sample_info.get("sample", {})
    failure_rollout_ids = [
        item.get("rollout_id")
        for item in evidence_rollouts
        if item.get("success") is not True and item.get("rollout_id")
    ]
    for correction_op in correction_ops:
        correction_op["task_tags"] = append_sample_metadata_tags(correction_op.get("task_tags") or [], sample)
        correction_op["source_failure_ids"] = failure_rollout_ids
        correction_op["recovery_source_ids"] = failure_rollout_ids
        normalized = _normalize_experience_operation(correction_op, default_kind="success", source=source)
        if normalized:
            norm_ops.append(normalized)
            print(
                f"  [Experience] Added supervised recovery candidate for {question_id} "
                "from failed training rollouts"
            )


def _write_experience_items(sample_dir, norm_ops):
    if not norm_ops:
        return
    with open(os.path.join(sample_dir, 'exp_items.json'), 'w', encoding='utf-8') as f:
        json.dump([_compact_experience_operation(op) for op in norm_ops], f, ensure_ascii=False, indent=2)


def _summarize_sample_trajectories(question_id, sample_dir, sample_rollout_results, args, llm, result):
    traj_paths = _trajectory_paths(sample_dir, sample_rollout_results, args)
    if not traj_paths:
        result['error'] = "No trajectory files found"
        print(f"  Warning: No trajectory files found for {question_id}")
        return None

    merged_summaries = summarize_rollouts(traj_paths, llm, sample_dir=sample_dir)
    if merged_summaries:
        return merged_summaries

    if not _trajectory_has_turns(traj_paths):
        result['error'] = "No turns in trajectories (empty trajectories)"
        print(f"  Warning: Skipping experience generation for {question_id} - trajectories contain no turns (only initial_prompt)")
    else:
        result['error'] = "Failed to generate summary"
        print(f"  Warning: Failed to generate summary for {question_id}")
    return None


def _generate_raw_experience_ops(
    *,
    sample_dir,
    sample_rollout_results,
    merged_summaries,
    args,
    llm,
):
    question, ground_truth, system_prompt_text, summaries_only = _summary_fields(
        merged_summaries, sample_rollout_results
    )
    used_experiences = _get_used_original_experiences(sample_dir)
    raw_ops = intra_sample_experiences(
        question,
        ground_truth,
        summaries_only,
        llm,
        max_ops=3,
        debug_dir=sample_dir,
        system_prompt=system_prompt_text,
        used_experiences=used_experiences,
    )
    return {
        "question": question,
        "ground_truth": ground_truth,
        "system_prompt_text": system_prompt_text,
        "summaries_only": summaries_only,
        "used_experiences": used_experiences,
        "raw_ops": raw_ops,
    }


def _normalize_sample_experience_ops(
    *,
    question_id,
    sample_info,
    sample_dir,
    sample_rollout_results,
    args,
    llm,
    raw_context,
):
    score_list = _rollout_scores(sample_rollout_results)
    evidence_rollouts = _evidence_rollouts(question_id, sample_dir, sample_rollout_results, args)
    success_state = _success_state(score_list)
    source = _experience_source(question_id, sample_info, score_list, evidence_rollouts)
    norm_ops = _normalize_llm_experience_ops(
        raw_context["raw_ops"],
        question_id,
        sample_info,
        source,
        success_state["default_kind"],
        success_state["has_success"],
        success_state["all_rollouts_failed"],
    )
    _add_auto_success_memory(
        norm_ops,
        question_id,
        sample_info,
        source,
        evidence_rollouts,
        success_state["has_success"],
    )
    _add_supervised_correction_ops(
        norm_ops,
        question_id,
        raw_context["question"],
        raw_context["ground_truth"],
        raw_context["summaries_only"],
        llm,
        sample_info,
        sample_dir,
        raw_context["system_prompt_text"],
        raw_context["used_experiences"],
        source,
        evidence_rollouts,
        success_state["all_rollouts_failed"],
    )
    return norm_ops


def generate_experience_for_sample(sample_info, args):
    """Generate normalized experience operations for one completed sample."""
    question_id = sample_info['question_id']
    sample_dir = sample_info['sample_dir']
    sample_rollout_results = sample_info['sample_rollout_results']

    result = _initial_generation_result(question_id)

    try:
        llm = ExperienceLLM()
        merged_summaries = _summarize_sample_trajectories(
            question_id, sample_dir, sample_rollout_results, args, llm, result
        )
        if not merged_summaries:
            return result

        raw_context = _generate_raw_experience_ops(
            sample_dir=sample_dir,
            sample_rollout_results=sample_rollout_results,
            merged_summaries=merged_summaries,
            args=args,
            llm=llm,
        )
        norm_ops = _normalize_sample_experience_ops(
            question_id=question_id,
            sample_info=sample_info,
            sample_dir=sample_dir,
            sample_rollout_results=sample_rollout_results,
            args=args,
            llm=llm,
            raw_context=raw_context,
        )

        _write_experience_items(sample_dir, norm_ops)
        result['success'] = True
        result['experience_ops'] = norm_ops

    except Exception as e:
        result['error'] = str(e)
        import traceback
        result['traceback'] = traceback.format_exc()

    return result
