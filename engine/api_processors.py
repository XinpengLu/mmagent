"""Run one API-based agent rollout and persist its trajectory."""

import os
import json
import re
import time
from qwen_vl_utils import fetch_image # type: ignore
from utils.image_utils import estimate_tokens, pil_to_base64_data_uri, process_image
from utils.judge import compute_score
from engine.api_caller import call_vision_api
from engine.api_model_caller import AgentState, create_model_caller, parse_function_call_response
from utils.artifacts import (
    INJECTED_SYSTEM_PROMPT_FILENAME,
    ROLLOUT_METRICS_FILENAME,
    clean_rollout_artifacts,
    save_trajectory,
)
from utils.transient import (
    text_has_transient_api_failure,
    text_has_transient_tool_failure,
)
from prompts.answer_templates import final_answer_synthesis_prompt

SYSTEM_PROMPT = ""
FINAL_ANSWER_MAX_TOKENS = 4096
FINAL_ANSWER_RETRIES = 3
TOOL_SYNTHESIS_AFTER_CALLS = 0
MAX_TURNS = 20
MAX_IMAGES = 50
MAX_TOTAL_TOKENS = 32768
MAX_PIXELS = 2_000_000
MIN_PIXELS = 40_000


def set_global_prompts(system_prompt):
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = system_prompt


def _question_and_image_paths(sample):
    return sample.get('problem', sample.get('question', '')), sample.get('images', [])


def _image_count_to_load(question, image_paths):
    placeholder_count = len(re.findall(r'<image>', question, re.IGNORECASE))
    if placeholder_count > 0:
        needed = placeholder_count
    elif image_paths:
        needed = len(image_paths)
    else:
        needed = 0
    return min(needed, len(image_paths)) if image_paths else 0, needed


def _initial_image_name(index):
    return f'original_image_{index}' if index > 0 else 'original_image'


def _initial_image_save_path(save_dir, index):
    filename = "original_image.jpg" if index == 0 else f"original_image_{index}.jpg"
    return os.path.join(save_dir, filename)


def _image_content_item(image, args):
    return {
        "type": "image_url",
        "image_url": {"url": pil_to_base64_data_uri(image, MAX_PIXELS, MIN_PIXELS)},
    }


def _load_initial_image(image_path, image_name, save_dir, index, args):
    original_image = fetch_image({'image': image_path, 'max_pixels': MAX_PIXELS})
    processed_image = process_image(original_image, MAX_PIXELS, MIN_PIXELS)
    os.makedirs(save_dir, exist_ok=True)
    processed_image.save(_initial_image_save_path(save_dir, index))
    return image_name, processed_image, _image_content_item(processed_image, args)


def _clean_question_text(question):
    question_text = re.sub(r'<image>', '', question, flags=re.IGNORECASE).strip()
    return re.sub(r'\n+', '\n', question_text).strip()


def _append_question_text(user_content_list, question_text):
    if question_text:
        user_content_list.append({"type": "text", "text": question_text})


def _parse_and_load_multiple_images(sample, args, save_dir):
    """Load the images referenced by a sample and build API user content."""
    question, image_paths = _question_and_image_paths(sample)
    num_images_to_load, num_images_needed = _image_count_to_load(question, image_paths)
    if num_images_to_load == 0 and num_images_needed > 0:
        print(f"Warning: Need {num_images_needed} image(s) but none provided in sample")
        return None, None, None, None

    image_map = {}
    image_list = []
    user_content_list = []
    for i in range(num_images_to_load):
        image_path = os.path.join(args.image_folder, image_paths[i])
        image_name = _initial_image_name(i)
        try:
            image_name, processed_image, content_item = _load_initial_image(
                image_path, image_name, save_dir, i, args
            )
        except FileNotFoundError:
            print(f"Error: Image not found at {image_path}")
            return None, None, None, None
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None, None, None, None
        image_map[image_name] = processed_image
        image_list.append((image_name, processed_image))
        user_content_list.append(content_item)

    question_text = _clean_question_text(question)
    _append_question_text(user_content_list, question_text)
    return image_map, image_list, question_text, user_content_list


def _initialize_sample_and_image(sample, args, save_dir):
    question_id = sample.get("doc_id", sample.get("question_id", "N/A"))
    question = sample.get('problem', sample.get('question', ''))
    ground_truth = sample.get("solution")

    initial_traj_info = {
        "doc_id": question_id,
        "initial_prompt": question,
        "ground_truth": ground_truth
    }
    save_trajectory(save_dir,initial_traj_info)

    image_map, image_list, question_text, user_content_list = _parse_and_load_multiple_images(sample, args, save_dir)
    if image_map is None:
        return None

    return question_id, question, ground_truth, image_map, image_list, question_text, user_content_list


def _create_initial_agent_state(question_text, image_map, image_list, user_content_list, system_prompt):
    """Create the initial text and multimodal histories for one rollout."""
    image_refs = [f"[Image: {name}]" for name, _ in image_list]
    user_content_text = "\n".join(image_refs) + (f"\n{question_text}" if question_text else "")

    if user_content_list:
        api_user_content = user_content_list
    else:
        api_user_content = [{"type": "text", "text": question_text}] if question_text else []

    return AgentState(
        conversation_history=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content_text}
        ],
        api_conversation_history=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": api_user_content
            }
        ],
        image_map=image_map.copy(),
        current_turn=0,
        current_token_count=estimate_tokens(system_prompt + question_text)
    )


def _extract_responses_from_node(node):
    conversation_history = node.conversation_history
    assistant_responses = [
        msg['content'] for msg in conversation_history
        if msg['role'] == 'assistant'
    ]
    return conversation_history, assistant_responses


def _evaluate_trajectory(question, ground_truth, conversation_history, assistant_responses, question_id):
    accuracy_score = 0.0
    trajectory_score = 0.0
    trajectory_analysis = ""
    trajectory_text = ""

    for message in conversation_history:
        trajectory_text += f"**{message['role']}**: {message['content']}\n\n"

    if os.environ.get("VERIFIER_API_KEY") and os.environ.get("VERIFIER_END_POINT"):
        if ground_truth:
            extra_info = {
                "acc_reward_weight": 1.0,
                "gpt_extract_answer": True,
                "extract_answer_tags": "strict",
            }
            try:
                accuracy_score, analysis = compute_score(
                    prompt=question,
                    predict_str_list=assistant_responses,
                    ground_truth=ground_truth,
                    extra_info=extra_info
                )
                trajectory_score = accuracy_score
                trajectory_analysis = analysis if analysis else "Evaluation completed."
            except Exception as e:
                print(f"Error during evaluation for {question_id}: {e}")
                accuracy_score = 0.0
                trajectory_score = 0.0
                trajectory_analysis = f"Evaluation failed: {str(e)}"
    else:
        print("VERIFIER_API_KEY or VERIFIER_END_POINT not found in environment variables. Skipping evaluation.")
        trajectory_analysis = "Evaluation skipped: VERIFIER_API_KEY or VERIFIER_END_POINT not configured."

    return accuracy_score, trajectory_score, trajectory_analysis, trajectory_text


def _build_result_dict(question_id, question, final_answer, ground_truth, conversation_history,
                       accuracy_score, trajectory_text, trajectory_score, trajectory_analysis):
    return {
        "question_id": question_id,
        "prompt": question,
        "final_answer": final_answer,
        "ground_truth": ground_truth,
        "conversation_history": conversation_history,
        "accuracy_score": accuracy_score,
        "trajectory_text": trajectory_text,
        "trajectory_score": trajectory_score,
        "trajectory_analysis": trajectory_analysis,
    }


def _mark_transient_failure(node, text):
    if text_has_transient_api_failure(text):
        setattr(node, "api_transient_failure", True)
        return "api_transient_failure"
    if text_has_transient_tool_failure(text):
        setattr(node, "tool_transient_failure", True)
        return "tool_transient_failure"
    return None


def _mandatory_tool_names(args):
    return [
        str(tool).strip()
        for tool in getattr(args, "_mandatory_tool_plan_tools", []) or []
        if str(tool).strip()
    ]


def _tool_call_name(tool_call):
    if not isinstance(tool_call, dict):
        return ""
    function = tool_call.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "").strip()
    return str(tool_call.get("name") or tool_call.get("tool_name") or "").strip()


def _has_executed_required_tool(node, mandatory_tool_names):
    required_tools = {name for name in mandatory_tool_names if name}
    for msg in node.api_conversation_history:
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls"):
            if not required_tools:
                return True
            for tool_call in msg.get("tool_calls") or []:
                if _tool_call_name(tool_call) in required_tools:
                    return True
    return False


def _count_executed_tool_calls(node):
    total = 0
    for msg in node.api_conversation_history:
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls"):
            total += len(msg.get("tool_calls") or [])
    return total


def _last_turn_executed_tool(node):
    for msg in reversed(node.api_conversation_history):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return bool(msg.get("tool_calls"))
    return False


def _save_turn_error(save_dir, turn_idx, text_output, error, error_type=None, traceback_text=None):
    event = {
        "turn_idx": turn_idx,
        "text_output": text_output,
        "error": error,
    }
    if error_type:
        event["error_type"] = error_type
    if traceback_text:
        event["traceback"] = traceback_text
    save_trajectory(save_dir, event)


def _save_final_synthesis_turn(save_dir, node, final_text, reason):
    save_trajectory(save_dir, {
        "turn_idx": getattr(node, "turn_offset", 0) + node.current_turn,
        "text_output": "[Final Answer Synthesis]\n" + final_text,
        "synthesis_reason": reason,
        "node_id": getattr(node, "node_id", ""),
    })


def _synthesize_final_answer(current_node, args, sampling_params, save_dir, reason="max_turns"):
    """Ask for one best-effort final answer without tools."""
    final_prompt = final_answer_synthesis_prompt(reason)
    messages = list(current_node.api_conversation_history)
    messages.append({"role": "user", "content": final_prompt})
    final_sampling_params = dict(sampling_params)
    final_sampling_params["max_tokens"] = min(
        int(final_sampling_params.get("max_tokens", FINAL_ANSWER_MAX_TOKENS)),
        FINAL_ANSWER_MAX_TOKENS,
    )

    try:
        response = call_vision_api(
            args.model_name,
            messages,
            final_sampling_params,
            max_retries=FINAL_ANSWER_RETRIES,
            tools=None,
        )
    except Exception as e:
        print(f"Warning: final answer synthesis failed: {e}")
        return False

    if isinstance(response, str) and response.startswith("Error:"):
        print(f"Warning: final answer synthesis API error: {response}")
        return False
    if response is None:
        print("Warning: final answer synthesis returned empty response.")
        return False

    response_text = response.get("content") if isinstance(response, dict) else response
    response_text = str(response_text or "").strip()
    if not response_text:
        return False

    action, data = parse_function_call_response(response_text)
    if action == "answer":
        final_text = response_text
        answer_value = data
    else:
        final_text = f"<answer>{response_text}</answer>"
        answer_value = response_text

    current_node.api_conversation_history.append({"role": "user", "content": final_prompt})
    current_node.conversation_history.append({"role": "user", "content": final_prompt})
    current_node.api_conversation_history.append({"role": "assistant", "content": final_text})
    current_node.conversation_history.append({"role": "assistant", "content": final_text})

    _save_final_synthesis_turn(save_dir, current_node, final_text, reason)
    current_node.current_turn += 1
    current_node.current_token_count += estimate_tokens(final_prompt + final_text)
    current_node.mark_final(answer_value)
    return True


def _stop_for_runtime_limits(node, args, sampling_params):
    if len(node.image_map) >= MAX_IMAGES:
        node.mark_final("Error: Reached max image limit.")
        return True
    if node.current_token_count >= MAX_TOTAL_TOKENS - sampling_params['max_tokens']:
        node.mark_final("Error: Reached max token limit.")
        return True
    return False


def _call_model_for_turn(current_node, model_caller, save_dir, turn):
    try:
        return model_caller(current_node)
    except Exception as e:
        print(f"Error calling model at turn {turn}: {e}")
        import traceback
        traceback.print_exc()
        error_msg = f"Error: Exception in model call: {str(e)}"
        error_type = _mark_transient_failure(current_node, error_msg) or "model_call_exception"
        current_node.mark_final(error_msg)
        _save_turn_error(
            save_dir,
            turn,
            error_msg,
            str(e),
            error_type=error_type,
            traceback_text=traceback.format_exc(),
        )
        return None


def _handle_model_error_response(current_node, response_text, save_dir, turn):
    if isinstance(response_text, str) and response_text.startswith("Error:"):
        error_msg = response_text
        print(f"Warning: {error_msg} at turn {turn}")
        current_node.mark_final(error_msg)
        error_type = _mark_transient_failure(current_node, error_msg) or "api_call_failed"
        _save_turn_error(
            save_dir,
            turn,
            error_msg,
            error_msg,
            error_type,
        )
        return True

    if response_text is None:
        error_msg = "Error: Empty response from model."
        print(f"Warning: {error_msg} at turn {turn}")
        current_node.mark_final(error_msg)
        error_type = _mark_transient_failure(current_node, error_msg) or "api_call_failed"
        _save_turn_error(save_dir, turn, error_msg, "Empty response from model", error_type)
        return True

    return False


def _maybe_synthesize_after_tool_budget(current_node, args, sampling_params, save_dir):
    tool_synthesis_after = TOOL_SYNTHESIS_AFTER_CALLS
    if tool_synthesis_after <= 0:
        return False
    if _count_executed_tool_calls(current_node) < tool_synthesis_after:
        return False
    return _synthesize_final_answer(current_node, args, sampling_params, save_dir, "tool_budget")


def _enforce_mandatory_tool_plan(current_node, mandatory_tool_names, save_dir):
    tools_text = ", ".join(mandatory_tool_names) if mandatory_tool_names else "the selected tool"
    feedback = (
        "A tool-backed check is required for this problem. "
        f"Before giving a final answer, execute at least one relevant tool call from: {tools_text}. "
        "Use the current image/problem to derive the tool input, inspect the tool output, then answer."
    )
    current_node.api_conversation_history.append({"role": "user", "content": feedback})
    current_node.conversation_history.append({"role": "user", "content": feedback})
    save_trajectory(save_dir, {
        "turn_idx": getattr(current_node, "turn_offset", 0) + current_node.current_turn,
        "text_output": "[Mandatory Tool Plan Enforcement] " + feedback,
        "node_id": getattr(current_node, "node_id", ""),
    })
    current_node.current_turn += 1


def _run_greedy_loop(current_node, model_caller, args, sampling_params, question_text, save_dir):
    """Run the turn loop until the agent answers or reaches a runtime stop."""
    final_answer = None
    mandatory_tool_required = bool(getattr(args, "_mandatory_tool_plan_required", False))
    mandatory_tool_names = _mandatory_tool_names(args)

    for turn in range(MAX_TURNS):
        if _stop_for_runtime_limits(current_node, args, sampling_params):
            break

        response_text = _call_model_for_turn(current_node, model_caller, save_dir, turn)
        if current_node.is_final and response_text is None:
            break

        if _handle_model_error_response(current_node, response_text, save_dir, turn):
            break

        if _last_turn_executed_tool(current_node):
            if _maybe_synthesize_after_tool_budget(current_node, args, sampling_params, save_dir):
                break
            continue

        action, data = parse_function_call_response(response_text)
        if action == "answer":
            if mandatory_tool_required and not _has_executed_required_tool(current_node, mandatory_tool_names):
                _enforce_mandatory_tool_plan(current_node, mandatory_tool_names, save_dir)
                continue
            final_answer = current_node.final_answer or data
            current_node.mark_final(final_answer)
            break

        if _synthesize_final_answer(
            current_node, args, sampling_params, save_dir, "parse_failure"
        ):
            break
        current_node.mark_final("Error: Could not parse model response.")
        break

    if current_node.is_final:
        final_answer = current_node.final_answer
    else:
        if not _synthesize_final_answer(current_node, args, sampling_params, save_dir, "max_turns"):
            final_answer = "Error: Reached max turns without a definitive answer."
            current_node.mark_final(final_answer)

    return current_node


def _api_usage_totals(node):
    api_events = getattr(node, "api_usage_events", []) or []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    api_elapsed = 0.0

    for event in api_events:
        usage = event.get("usage") if isinstance(event, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
        prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        total_tokens += int(usage.get("total_tokens", 0) or 0)
        try:
            api_elapsed += float(event.get("elapsed_seconds") or 0.0)
        except Exception:
            pass

    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return api_events, prompt_tokens, completion_tokens, total_tokens, api_elapsed


def _update_result_runtime_metrics(result, final_node, args, sampling_params, runtime_seconds, trajectory_text):
    api_transient_failure = bool(getattr(final_node, "api_transient_failure", False))
    if not api_transient_failure:
        api_transient_failure = text_has_transient_api_failure(result.get("final_answer"))
    tool_transient_failure = bool(getattr(final_node, "tool_transient_failure", False))
    if not tool_transient_failure:
        tool_transient_failure = text_has_transient_tool_failure(result.get("final_answer")) or text_has_transient_tool_failure(trajectory_text)

    failure_type = None
    if api_transient_failure:
        failure_type = "api_transient_failure"
    elif tool_transient_failure:
        failure_type = "tool_transient_failure"

    api_events, prompt_tokens, completion_tokens, total_tokens, api_elapsed = _api_usage_totals(final_node)
    result.update({
        "api_transient_failure": api_transient_failure,
        "tool_transient_failure": tool_transient_failure,
        "failure_type": failure_type,
        "reasoning_model_name": getattr(args, "model_name", None),
        "runtime_seconds": round(runtime_seconds, 4),
        "api_elapsed_seconds": round(api_elapsed, 4),
        "api_call_count": len(api_events),
        "api_prompt_tokens": prompt_tokens,
        "api_completion_tokens": completion_tokens,
        "api_total_tokens": total_tokens,
        "estimated_context_tokens": getattr(final_node, "current_token_count", 0),
        "turn_count": getattr(final_node, "current_turn", 0),
        "tool_call_count": _count_executed_tool_calls(final_node),
        "sampling_temperature": sampling_params.get("temperature"),
        "sampling_top_p": sampling_params.get("top_p"),
    })


def _write_injected_system_prompt(save_dir, system_prompt):
    try:
        with open(os.path.join(save_dir, INJECTED_SYSTEM_PROMPT_FILENAME), 'w', encoding='utf-8') as f:
            f.write(system_prompt)
    except Exception:
        pass


def _run_model_with_cleanup(initial_node, model_caller, args, sampling_params, question_text, save_dir):
    try:
        return _run_greedy_loop(
            initial_node, model_caller, args, sampling_params, question_text, save_dir
        )
    finally:
        close_tools = getattr(model_caller, "close_tools", None)
        if callable(close_tools):
            close_tools()


def _evaluate_final_node(question_id, question_text, ground_truth, final_node):
    conversation_history, assistant_responses = _extract_responses_from_node(final_node)
    accuracy_score, trajectory_score, trajectory_analysis, trajectory_text = _evaluate_trajectory(
        question_text, ground_truth, conversation_history, assistant_responses, question_id
    )
    result = _build_result_dict(
        question_id,
        question_text,
        final_node.final_answer,
        ground_truth,
        conversation_history,
        accuracy_score,
        trajectory_text,
        trajectory_score,
        trajectory_analysis,
    )
    return result, trajectory_text


def _process_single_sample_unified(sample, args, sampling_params, save_dir=None):
    """Run one sample and return its rollout result."""
    if save_dir is None:
        save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    rollout_started_at = time.time()

    init_result = _initialize_sample_and_image(sample, args, save_dir)
    if init_result is None:
        return None
    question_id, question, ground_truth, image_map, image_list, question_text, user_content_list = init_result

    current_system_prompt = getattr(args, '_sample_system_prompt', SYSTEM_PROMPT)
    _write_injected_system_prompt(save_dir, current_system_prompt)
    initial_node = _create_initial_agent_state(question_text, image_map, image_list, user_content_list, current_system_prompt)

    model_caller = create_model_caller(args, sampling_params, save_dir)
    final_node = _run_model_with_cleanup(initial_node, model_caller, args, sampling_params, question_text, save_dir)
    runtime_seconds = time.time() - rollout_started_at

    result, trajectory_text = _evaluate_final_node(question_id, question_text, ground_truth, final_node)
    _update_result_runtime_metrics(result, final_node, args, sampling_params, runtime_seconds, trajectory_text)

    return result


def process_single_sample(sample, args, sampling_params, rollout_idx=None):
    """Process one sample and save its rollout artifacts."""
    question_id = sample.get("doc_id", sample.get("question_id", "N/A"))

    if rollout_idx is not None and args.rollouts_per_sample > 1:
        sample_dir = os.path.join(args.output_dir, question_id)
        save_dir = os.path.join(sample_dir, f"rollout_{rollout_idx}")
    else:
        save_dir = os.path.join(args.output_dir, question_id)

    os.makedirs(save_dir, exist_ok=True)
    clean_rollout_artifacts(save_dir)

    try:
        final_result = _process_single_sample_unified(sample, args, sampling_params, save_dir=save_dir)

        if final_result:
            metrics = {
                "accuracy_score": final_result["accuracy_score"],
                "trajectory_score": final_result.get("trajectory_score"),
                "trajectory_analysis": final_result.get("trajectory_analysis"),
                "reasoning_model_name": final_result.get("reasoning_model_name"),
                "runtime_seconds": final_result.get("runtime_seconds"),
                "api_elapsed_seconds": final_result.get("api_elapsed_seconds"),
                "api_call_count": final_result.get("api_call_count"),
                "api_prompt_tokens": final_result.get("api_prompt_tokens"),
                "api_completion_tokens": final_result.get("api_completion_tokens"),
                "api_total_tokens": final_result.get("api_total_tokens"),
                "estimated_context_tokens": final_result.get("estimated_context_tokens"),
                "turn_count": final_result.get("turn_count"),
                "tool_call_count": final_result.get("tool_call_count"),
                "sampling_temperature": final_result.get("sampling_temperature"),
                "sampling_top_p": final_result.get("sampling_top_p"),
                "api_transient_failure": bool(final_result.get("api_transient_failure")),
                "tool_transient_failure": bool(final_result.get("tool_transient_failure")),
                "failure_type": final_result.get("failure_type"),
            }
            with open(os.path.join(save_dir, ROLLOUT_METRICS_FILENAME), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)

        return final_result

    except Exception as e:
        print(f"[Error] Failed to process sample: {e}")
        raise
