"""OpenAI-compatible chat API caller used by the rollout engine."""

import os
import time
import random
import json
import requests
from utils.api_usage_logger import log_api_usage
from utils.key_pool import llm_key_should_rotate, rotate_current_key

NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404}
API_TIMEOUT_SECONDS = 240
API_BASE_WAIT_SECONDS = 1.0
API_MAX_WAIT_SECONDS = 30.0
API_HIGH_TOKEN_WARNING_THRESHOLD = 25000
API_MAX_RETRIES = 3


def _runtime_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _is_openai_reasoning_model(model_name: str) -> bool:
    """Return True for OpenAI o-series / GPT-5 reasoning models."""
    model_id = (model_name or "").lower().split("/")[-1]
    return model_id.startswith(("o1", "o3", "o4", "gpt-5"))


def _build_payload(model_name: str, messages: list, sampling_params: dict, tools: list = None):
    payload = {
        "model": model_name,
        "messages": messages,
    }

    if _is_openai_reasoning_model(model_name):
        payload["max_completion_tokens"] = sampling_params['max_tokens']
    else:
        payload["temperature"] = sampling_params['temperature']
        payload["top_p"] = sampling_params['top_p']
        payload["max_tokens"] = sampling_params['max_tokens']
    
    if tools:
        payload["tools"] = tools
        if "o4-mini" not in model_name:
            payload["parallel_tool_calls"] = False
    
    return payload


def _add_reasoning_param(payload: dict, model_name: str, end_point: str = None):
    model_lower = model_name.lower()
    is_gemini_model = "gemini" in model_lower
    is_gpt_model = _is_openai_reasoning_model(model_name)
    
    is_openrouter = False
    if end_point:
        is_openrouter = "openrouter" in end_point.lower()
    
    if is_gemini_model:
        if is_openrouter:
            reasoning_max_tokens = os.environ.get("REASONING_MAX_TOKENS")
            if reasoning_max_tokens and reasoning_max_tokens.lower() not in ["none", "false", ""]:
                payload["reasoning"] = {
                    "max_tokens": int(reasoning_max_tokens),
                    "exclude": False
                }
        else:
            reasoning_effort = os.environ.get("REASONING_EFFORT")
            if reasoning_effort and reasoning_effort.lower() not in ["none", "false", ""]:
                payload["reasoning"] = {
                    "effort": reasoning_effort,
                    "exclude": False
                }
    elif is_gpt_model:
        reasoning_effort = os.environ.get("REASONING_EFFORT")
        if reasoning_effort and reasoning_effort.lower() not in ["none", "false", ""]:
            payload["reasoning_effort"] = reasoning_effort
    else:
        pass


def _make_api_request(end_point: str, headers: dict, payload: dict, api_name: str = "API"):
    try:
        response = requests.post(
            end_point,
            headers=headers,
            json=payload,
            timeout=_runtime_int("API_TIMEOUT_SECONDS", API_TIMEOUT_SECONDS),
        )
        return response, None
    except requests.exceptions.Timeout:
        print(f"[{api_name}] API timeout")
        return None, "timeout"
    except requests.exceptions.RequestException as e:
        print(f"[{api_name}] API call failed: {e}")
        return None, "network"
    except Exception as e:
        print(f"[{api_name}] Unexpected error: {e}")
        return None, "other"


def _response_json(response):
    if response is None:
        return None
    try:
        return response.json()
    except Exception:
        return None


def _log_reasoning_usage(api_name, model_name, end_point, payload, response, request_error, attempt):
    response_data = _response_json(response)
    log_api_usage(
        source=api_name,
        model=model_name,
        endpoint=end_point,
        payload=payload,
        response_data=response_data,
        status_code=getattr(response, "status_code", None),
        elapsed_seconds=_response_elapsed_seconds(response),
        success=bool(response is not None and response.status_code == 200 and not request_error),
        error=request_error,
        attempt=attempt + 1,
    )


def _retry_wait(attempt: int) -> float:
    """Exponential backoff with jitter for transient API failures."""
    wait_time = API_BASE_WAIT_SECONDS * (2 ** attempt) + random.uniform(0, 1)
    return min(wait_time, API_MAX_WAIT_SECONDS)


def _max_retry_count(value) -> int:
    return max(1, int(value))


def _api_error_info(response, api_name: str, error_type: str = None) -> str:
    if error_type == "http_error":
        if response is None:
            return f"{api_name}: http_error"
        try:
            error_detail = response.json() if response else {}
            message = error_detail.get("error", {}).get("message", str(error_detail))
            return f"{api_name}: HTTP {response.status_code} - {message}"
        except Exception:
            return f"{api_name}: HTTP {response.status_code}"
    return f"{api_name}: {error_type}" if error_type else f"{api_name}: unknown_error"


def _response_error_message(response) -> str:
    if response is None:
        return ""
    try:
        detail = response.json()
    except Exception:
        return getattr(response, "text", "")
    if isinstance(detail, dict):
        error = detail.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        return str(detail)
    return str(detail)


def _attempt_suffix(attempt: int, max_attempts: int) -> str:
    return f" on attempt {attempt + 1}/{max_attempts}" if max_attempts > 1 else ""


def _parse_http_error(response, api_name: str, attempt: int, max_attempts: int):
    suffix = _attempt_suffix(attempt, max_attempts)
    try:
        detail = response.json()
    except Exception:
        detail = response.text
    print(f"[{api_name}] API Error {response.status_code}{suffix}: {detail}")
    return None, response.status_code == 429, "http_error"


def _response_elapsed_seconds(response):
    try:
        return response.elapsed.total_seconds()
    except Exception:
        return None


def _message_from_choice(result: dict, api_name: str, suffix: str):
    first_choice = result["choices"][0]
    if first_choice is None:
        print(f"[{api_name}] API returned None in choices[0]{suffix}")
        return None
    if not isinstance(first_choice, dict) or "message" not in first_choice:
        print(f"[{api_name}] API response missing 'message' field in choices[0]{suffix}")
        print(f"[{api_name}] choices[0] content: {first_choice}")
        return None
    message = first_choice["message"]
    if message is None or not isinstance(message, dict):
        print(f"[{api_name}] API returned invalid message type{suffix}: {type(message)}")
        return None
    return message


def _attach_response_metadata(message: dict, result: dict, response) -> None:
    usage = result.get("usage", {}) if isinstance(result, dict) else {}
    elapsed_seconds = _response_elapsed_seconds(response)
    if usage:
        message["_api_usage"] = usage
    if elapsed_seconds is not None:
        message["_api_elapsed_seconds"] = elapsed_seconds


def _parse_choice_message(result: dict, response, api_name: str, attempt: int, max_attempts: int):
    suffix = _attempt_suffix(attempt, max_attempts)
    message = _message_from_choice(result, api_name, suffix)
    if message is None:
        return None, False, "invalid_format"

    _attach_response_metadata(message, result, response)
    if message.get("tool_calls"):
        return message, False, None
    if "tool_calls" in message:
        message.pop("tool_calls", None)

    content = message.get("content")
    if content and content.strip():
        return message, False, None

    print(f"[{api_name}] API returned empty content{suffix}")
    if max_attempts > 1:
        try:
            print(f"[{api_name}] Empty message detail: {json.dumps(message, ensure_ascii=False)}")
        except Exception:
            print(f"[{api_name}] Empty message detail: {message}")
    return None, False, "empty_content"


def _parse_empty_or_invalid_choices(result: dict, api_name: str, attempt: int, max_attempts: int):
    suffix = _attempt_suffix(attempt, max_attempts)
    choices = result.get("choices")
    if not isinstance(choices, list) or len(choices) != 0:
        print(f"[{api_name}] Invalid API response format{suffix}: {result}")
        return None, False, "invalid_format"

    print(f"[{api_name}] Model refused to generate (empty choices){suffix}")
    usage = result.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    print(f"[{api_name}] Usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")
    if "error" in result:
        print(f"[{api_name}] Error details: {result['error']}")
    if prompt_tokens > API_HIGH_TOKEN_WARNING_THRESHOLD:
        print(f"[{api_name}] Warning: Very high prompt_tokens ({prompt_tokens}), may have hit token limit")
    print(f"[{api_name}] Possible causes: content filtering, safety policy, token limit, or model overload")
    return None, False, "empty_choices"


def _parse_api_response(response, api_name: str, attempt: int = 0, max_attempts: int = 1):
    if response.status_code != 200:
        return _parse_http_error(response, api_name, attempt, max_attempts)

    try:
        result = response.json()
    except json.JSONDecodeError as e:
        print(f"[{api_name}] Failed to parse JSON response: {e}")
        return None, False, "invalid_format"

    if "choices" in result and result["choices"]:
        return _parse_choice_message(result, response, api_name, attempt, max_attempts)
    return _parse_empty_or_invalid_choices(result, api_name, attempt, max_attempts)

def _api_headers(api_key: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def _retry_after_request_error(api_name: str, request_error: str, attempt: int, max_retries: int):
    last_error = f"{api_name}: {request_error}"
    if attempt >= max_retries - 1:
        return False, last_error
    time.sleep(_retry_wait(attempt))
    return True, last_error


def _retry_after_rate_limit(api_name: str, attempt: int, max_retries: int):
    last_error = f"{api_name}: HTTP 429 (rate limit)"
    if attempt >= max_retries - 1:
        return False, last_error
    wait_time = _retry_wait(attempt)
    print(
        f"[{api_name}] API rate limit hit on attempt {attempt + 1}/{max_retries}. "
        f"Waiting {wait_time:.2f} seconds before retrying."
    )
    time.sleep(wait_time)
    return True, last_error


def _retry_after_parse_error(response, api_name: str, error_type: str, attempt: int, max_retries: int):
    last_error = _api_error_info(response, api_name, error_type)
    if error_type == "empty_choices":
        return False, last_error
    if error_type == "http_error" and response and response.status_code in NON_RETRYABLE_STATUS_CODES:
        return False, last_error
    if attempt >= max_retries - 1:
        return False, last_error

    wait_time = _retry_wait(attempt)
    if error_type == "http_error" and response is not None and response.status_code >= 500:
        print(f"[{api_name}] Transient server error; waiting {wait_time:.2f}s before retrying.")
    time.sleep(wait_time)
    return True, last_error


def _try_single_api(api_key: str, end_point: str, model_name: str, messages: list, 
                    sampling_params: dict, max_retries: int, api_name: str = "API", tools: list = None):
    payload = _build_payload(model_name, messages, sampling_params, tools)
    _add_reasoning_param(payload, model_name, end_point)
    last_error = None
    
    for attempt in range(max_retries):
        headers = _api_headers(api_key)
        response, request_error = _make_api_request(end_point, headers, payload, api_name)
        _log_reasoning_usage(api_name, model_name, end_point, payload, response, request_error, attempt)
        if request_error:
            should_retry, last_error = _retry_after_request_error(api_name, request_error, attempt, max_retries)
            if should_retry:
                continue
            return None, last_error

        if response.status_code == 429:
            if llm_key_should_rotate(response.status_code, _response_error_message(response)):
                api_key = rotate_current_key("reasoning", api_key)
                continue
            should_retry, last_error = _retry_after_rate_limit(api_name, attempt, max_retries)
            if should_retry:
                continue
            return None, last_error

        result, is_429, error_type = _parse_api_response(response, api_name, attempt=attempt, max_attempts=max_retries)
        if result is not None:
            return result, None

        if error_type == "http_error" and llm_key_should_rotate(response.status_code, _response_error_message(response)):
            api_key = rotate_current_key("reasoning", api_key)
            continue

        should_retry, last_error = _retry_after_parse_error(response, api_name, error_type, attempt, max_retries)
        if should_retry:
            continue
        return None, last_error
    
    return None, last_error


def _reasoning_api_config():
    config = {
        "api_key": os.environ.get("REASONING_API_KEY"),
        "end_point": os.environ.get("REASONING_END_POINT"),
        "api_name": "Reasoning API",
    }
    if not all([config["api_key"], config["end_point"]]):
        raise ValueError("REASONING_API_KEY and REASONING_END_POINT must be set.")
    return config


def _call_reasoning_api(config, model_name, messages, sampling_params, max_retries, tools):
    result, error_info = _try_single_api(
        config["api_key"],
        config["end_point"],
        model_name,
        messages,
        sampling_params,
        max_retries,
        api_name=config["api_name"],
        tools=tools,
    )
    if result is not None:
        return result

    error_summary = f"Reasoning API failed after {max_retries} attempts"
    if error_info:
        error_summary += f": {error_info}"
    print(f"[API] {error_summary}")
    return f"Error: {error_summary}"


def call_vision_api(model_name: str, messages: list, sampling_params: dict, max_retries: int = None, tools: list = None):
    """Call the reasoning API with bounded retries."""
    if max_retries is None:
        max_retries = _runtime_int("API_MAX_RETRIES", API_MAX_RETRIES)
    max_retries = _max_retry_count(max_retries)
    config = _reasoning_api_config()
    return _call_reasoning_api(config, model_name, messages, sampling_params, max_retries, tools)

