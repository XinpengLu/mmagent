"""Transient rollout failure detection shared by rollout and engine code."""

from __future__ import annotations

from typing import Any, Dict, Optional


TRANSIENT_FAILURE_TYPES = {"api_transient_failure", "tool_transient_failure"}

TRANSIENT_API_FAILURE_MARKERS = (
    "all api attempts failed",
    "reasoning api failed after",
    "empty_content",
    "empty choices",
    "empty_choices",
    "empty response from model",
    "model refused to generate",
    "api_transient_failure",
    "invalid_prompt",
    "invalid prompt",
    "content_policy",
    "content policy",
    "policy violation",
)

TRANSIENT_TOOL_FAILURE_MARKERS = (
    "kernel died before replying to kernel_info",
    "failed to initialize kernel client",
    "failed to start kernel",
    "kernel connection file not created",
    "kernel is not ready",
)


def text_has_transient_api_failure(text: Any) -> bool:
    if text is None:
        return False
    lowered = str(text).lower()
    return any(marker in lowered for marker in TRANSIENT_API_FAILURE_MARKERS)


def text_has_transient_tool_failure(text: Any) -> bool:
    if text is None:
        return False
    lowered = str(text).lower()
    return any(marker in lowered for marker in TRANSIENT_TOOL_FAILURE_MARKERS)


def text_has_transient_failure(text: Any) -> bool:
    return text_has_transient_api_failure(text) or text_has_transient_tool_failure(text)


def result_has_transient_failure(result: Dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    return bool(
        result.get("api_transient_failure")
        or result.get("tool_transient_failure")
        or result.get("failure_type") in TRANSIENT_FAILURE_TYPES
    )


def rollout_dir_has_transient_failure(rollout_dir: str, metrics: Optional[Dict[str, Any]] = None) -> bool:
    return bool(isinstance(metrics, dict) and (
        metrics.get("api_transient_failure")
        or metrics.get("tool_transient_failure")
        or metrics.get("failure_type") in TRANSIENT_FAILURE_TYPES
    ))
