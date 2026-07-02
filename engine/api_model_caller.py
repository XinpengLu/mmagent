"""
Model caller for API-based inference (vision API + function calling).
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from utils.artifacts import save_trajectory
from engine.api_caller import call_vision_api
from engine.api_tool_handler import APIToolHandler, InvalidToolCallError

CALLABLE_TOOL_NAMES = ("code_interpreter", "web_search", "visit", "image_search", "zoom")


@dataclass
class AgentState:
    """Mutable state for one multi-step reasoning trajectory."""

    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    api_conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    image_map: Dict[str, Image.Image] = field(default_factory=dict)
    current_turn: int = 0
    current_token_count: int = 0
    is_final: bool = False
    final_answer: Optional[str] = None
    node_id: str = ""
    created_at: float = field(default_factory=time.time)
    api_usage_events: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.node_id:
            self.node_id = f"state_{int(time.time() * 1000000)}"

    def mark_final(self, answer: str) -> None:
        self.is_final = True
        self.final_answer = answer


def _build_openai_tools_schema(tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Build OpenAI function-calling schemas from the registered tools."""
    from tools import TOOL_REGISTRY, get_tool_info

    names = tool_names if tool_names is not None else list(TOOL_REGISTRY.keys())
    tools = []
    for tool_name in names:
        try:
            tool_info = get_tool_info(tool_name)
            if not tool_info:
                print(f"Warning: Tool '{tool_name}' not found in registry, skipping")
                continue
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool_info["name"],
                    "description": tool_info["description"],
                },
            }
            if tool_info.get("parameters"):
                tool_def["function"]["parameters"] = tool_info["parameters"]
            tools.append(tool_def)
        except Exception as exc:
            print(f"Warning: Failed to build schema for tool '{tool_name}': {exc}")
    return tools


def _parse_inline_tool_arguments(raw_name: str) -> dict[str, Any]:
    params = {}
    inline_text = raw_name
    match = re.search(r"\((.*?)\)", raw_name, flags=re.S)
    if match:
        inline_text = match.group(1)
    for key, quote, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(['\"])(.*?)\2", inline_text, flags=re.S):
        params[key] = value
    for key, quote, value in re.findall(r'["\']([A-Za-z_][A-Za-z0-9_]*)["\']\s*:\s*(["\'])(.*?)\2', inline_text, flags=re.S):
        params[key] = value
    return params


def _normalize_tool_call(tool_name: str, parameters: Any) -> tuple[str, Any]:
    raw_name = str(tool_name or "").strip()
    if raw_name in CALLABLE_TOOL_NAMES:
        return raw_name, parameters
    for name in CALLABLE_TOOL_NAMES:
        if re.match(rf'^{re.escape(name)}\s*(?:\(|<|,|"|\'|$)', raw_name):
            fixed_parameters = parameters or _parse_inline_tool_arguments(raw_name)
            print(f"[Function Call Parser] Normalized malformed tool name '{raw_name}' -> '{name}'")
            return name, fixed_parameters
    return raw_name, parameters


def parse_function_call_response(response: Union[str, Dict], text_content: str = None) -> Tuple[str, Any]:
    if isinstance(response, dict):
        tool_calls = response.get("tool_calls") or []
        if tool_calls:
            if len(tool_calls) > 1:
                print("[Function Call Parser] Warning: Multiple tool calls detected, using only the first one")
            first_call = tool_calls[0]
            function_data = first_call.get("function", {})
            tool_name = function_data.get("name", "")
            arguments = function_data.get("arguments", "{}")
            if isinstance(arguments, str):
                try:
                    parameters = json.loads(arguments)
                except json.JSONDecodeError:
                    return "error", f"Invalid JSON in arguments: {arguments}"
            else:
                parameters = arguments
            if not tool_name:
                return "error", "Tool call missing 'name' field"
            tool_name, parameters = _normalize_tool_call(tool_name, parameters)
            return "function_call", {
                "tool_name": tool_name,
                "parameters": parameters,
                "tool_call_id": first_call.get("id"),
            }

        response = text_content or response.get("content") or response.get("text") or ""

    if isinstance(response, str):
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if answer_match:
            return "answer", answer_match.group(1).strip()
        return "text", response

    return "error", f"Unknown response format: {type(response)}"


def _extract_reasoning_text(reasoning_details):
    """Extract display text from provider-specific reasoning metadata."""
    if not reasoning_details:
        return None
    
    if isinstance(reasoning_details, str):
        return reasoning_details
    
    if isinstance(reasoning_details, list):
        texts = []
        for detail in reasoning_details:
            if isinstance(detail, dict):
                if "text" in detail:
                    texts.append(detail["text"])
                elif "summary" in detail:
                    texts.append(detail["summary"])
                elif "content" in detail:
                    texts.append(str(detail["content"]))
            elif isinstance(detail, str):
                texts.append(detail)
        return "\n".join(texts) if texts else None
    
    if isinstance(reasoning_details, dict):
        if "text" in reasoning_details:
            return reasoning_details["text"]
        elif "content" in reasoning_details:
            return str(reasoning_details["content"])
        elif "summary" in reasoning_details:
            return reasoning_details["summary"]
    
    return None


def _build_tools_schema(enabled_tools: str = ""):
    enable_fc = os.environ.get("ENABLE_FUNCTION_CALLING", "true")
    if enable_fc.lower() != "true":
        print(f"[Function Calling] Disabled (ENABLE_FUNCTION_CALLING={enable_fc})")
        return None

    try:
        from tools import list_tools

        all_tools = list_tools()
        enabled_tools = str(enabled_tools or "").strip()
        if enabled_tools:
            enabled_tools_list = [tool.strip() for tool in enabled_tools.split(",") if tool.strip()]
            tool_names = [tool for tool in enabled_tools_list if tool in all_tools]
            missing_tools = [tool for tool in enabled_tools_list if tool not in all_tools]
            if missing_tools:
                print(f"[Function Calling] Warning: enabled tools not found in registry: {missing_tools}")
        else:
            tool_names = all_tools
            print(f"[Function Calling] No enabled tools configured, using all registered tools: {tool_names}")

        return _build_openai_tools_schema(tool_names)
    except Exception as e:
        print(f"[Function Calling] Warning: Failed to build tools schema: {e}")
        return None


def _response_text(response):
    if isinstance(response, dict):
        return response.get("content") or ""
    return response if isinstance(response, str) else ""


def _record_api_usage(node, response, sampling_params):
    if not isinstance(response, dict):
        return

    api_usage = response.pop("_api_usage", None)
    api_elapsed_seconds = response.pop("_api_elapsed_seconds", None)
    if not api_usage and api_elapsed_seconds is None:
        return

    event = {
        "turn": getattr(node, "current_turn", 0),
        "usage": api_usage or {},
        "elapsed_seconds": api_elapsed_seconds,
        "temperature": sampling_params.get("temperature"),
        "top_p": sampling_params.get("top_p"),
    }
    if not hasattr(node, "api_usage_events"):
        node.api_usage_events = []
    node.api_usage_events.append(event)


def _extract_response_reasoning(response, response_text):
    reasoning_details = None
    reasoning_text = None

    if not isinstance(response, dict):
        return reasoning_details, reasoning_text

    if "reasoning_content" in response:
        reasoning_text = response.get("reasoning_content")
        if not isinstance(reasoning_text, str) or not reasoning_text.strip():
            reasoning_text = None

    if not reasoning_text:
        reasoning_details = response.get("reasoning_details") or response.get("reasoning")
        if reasoning_details:
            reasoning_text = _extract_reasoning_text(reasoning_details)

    if not reasoning_text and isinstance(response.get("reasoning"), str):
        reasoning_text = response.get("reasoning")

    if "tool_calls" in response and not reasoning_text and response_text and response_text.strip():
        reasoning_text = response_text.strip()
        print(f"[Reasoning] Using content field as reasoning_text: {len(reasoning_text)} chars")

    return reasoning_details, reasoning_text


def _first_tool_call_for_history(tool_calls_list, node):
    first_tool_call_only = [dict(tool_calls_list[0])] if tool_calls_list else []
    if first_tool_call_only and not first_tool_call_only[0].get("id"):
        turn_offset = getattr(node, 'turn_offset', 0)
        first_tool_call_only[0]["id"] = f"call_{turn_offset + node.current_turn}"
    return first_tool_call_only


def _append_assistant_response(node, response, response_text, reasoning_details):
    is_tool_call_response = isinstance(response, dict) and bool(response.get("tool_calls"))
    tool_calls_list = response.get("tool_calls", []) if is_tool_call_response else []

    if is_tool_call_response:
        assistant_msg = {
            "role": "assistant",
            "content": response_text,
            "tool_calls": _first_tool_call_for_history(tool_calls_list, node),
        }
        node.conversation_history.append({"role": "assistant", "content": f"[Function Call] {response_text}"})
    else:
        assistant_msg = {"role": "assistant", "content": response_text}
        node.conversation_history.append({"role": "assistant", "content": response_text})

    if reasoning_details:
        assistant_msg["reasoning_details"] = reasoning_details
        kind = "tool call" if is_tool_call_response else "text"
        print(f"[Reasoning] Preserved reasoning_details in {kind} message")
    node.api_conversation_history.append(assistant_msg)


def _global_turn_idx(node):
    return getattr(node, 'turn_offset', 0) + node.current_turn


def _tool_names_for_display(response):
    if not isinstance(response, dict) or "tool_calls" not in response:
        return []
    tool_names = []
    for tool_call in response.get("tool_calls", [])[:1]:
        func_info = tool_call.get("function", {})
        tool_names.append(func_info.get("name", "unknown"))
    return tool_names


def _trajectory_display_text(response, response_text, reasoning_text):
    tool_names = _tool_names_for_display(response)
    if not tool_names:
        return reasoning_text if reasoning_text else response_text

    display_parts = []
    if reasoning_text and reasoning_text.strip():
        display_parts.append(reasoning_text.strip())
    elif response_text and response_text.strip():
        display_parts.append(response_text.strip())
    display_parts.append(f"\n[Calling tools: {', '.join(tool_names)}]")
    return "\n\n".join(display_parts)


def _save_model_turn(save_dir, node, global_turn_idx, response, response_text, reasoning_text):
    if not save_dir:
        return
    save_trajectory(
        save_dir,
        {
            "turn_idx": global_turn_idx,
            "text_output": _trajectory_display_text(response, response_text, reasoning_text),
            "node_id": getattr(node, 'node_id', ''),
        },
    )


def _tool_call_id_from_history(node):
    if not node.api_conversation_history:
        return None
    last_msg = node.api_conversation_history[-1]
    if not isinstance(last_msg, dict) or last_msg.get("role") != "assistant":
        return None
    tool_calls = last_msg.get("tool_calls") or []
    return tool_calls[0].get("id", "call_0") if tool_calls else None


def _handle_model_action(action, data, node, tool_handler, global_turn_idx):
    if action == "function_call" or action == "tool_call":
        tool_name = data.get("tool_name", "")
        parameters = data.get("parameters", {})
        tool_call_id = data.get("tool_call_id") or _tool_call_id_from_history(node)

        try:
            result = tool_handler.execute_tool_call(
                tool_name, parameters, node, global_turn_idx, tool_call_id
            )
        except InvalidToolCallError as exc:
            node.tool_transient_failure = True
            node.mark_final(f"Error: tool_transient_failure: {exc}")
            return "invalid_tool_call"
        if getattr(node, "tool_transient_failure", False):
            return "invalid_tool_call"
        if result.get('feedback_messages'):
            node.api_conversation_history.extend(result['feedback_messages'])
        return bool(result.get('skip_processing'))

    if action == "answer":
        node.final_answer = data
    return False


def create_model_caller(args, sampling_params, save_dir=None):
    """Create the callable used by the greedy rollout loop."""
    tool_handler = APIToolHandler(args, save_dir)
    tools_schema = _build_tools_schema(args.enabled_tools)
    
    def model_caller(node: AgentState):
        try:
            messages = node.api_conversation_history.copy()
            response = call_vision_api(args.model_name, messages, sampling_params, tools=tools_schema)

            if isinstance(response, str) and response.startswith("Error:"):
                print(f"[Model Caller] API returned error: {response}")
                return response

            if response is None:
                return None

            _record_api_usage(node, response, sampling_params)
            response_text = _response_text(response)
            reasoning_details, reasoning_text = _extract_response_reasoning(response, response_text)
            _append_assistant_response(node, response, response_text, reasoning_details)

            global_turn_idx = _global_turn_idx(node)
            _save_model_turn(save_dir, node, global_turn_idx, response, response_text, reasoning_text)

            action, data = parse_function_call_response(response, text_content=response_text)
            action_result = _handle_model_action(action, data, node, tool_handler, global_turn_idx)
            if action_result == "invalid_tool_call":
                return None
            if action_result:
                node.current_turn += 1
                return response_text

            node.current_turn += 1
            return response_text
            
        except Exception as e:
            import traceback
            print(f"Error in model caller: {e}")
            traceback.print_exc()
            return None

    model_caller.close_tools = tool_handler.close
    return model_caller
