"""Lightweight JSONL API usage logging for experiment bookkeeping."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


_LOCK = threading.Lock()
_RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_PATH: Path | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def usage_log_path() -> Path:
    global _LOG_PATH
    if _LOG_PATH is None:
        log_dir = _project_root() / "api_usage"
        log_dir.mkdir(parents=True, exist_ok=True)
        _LOG_PATH = log_dir / f"api_usage_{_RUN_STAMP}_{os.getpid()}.jsonl"
    return _LOG_PATH


def _usage(response_data: dict[str, Any] | None) -> dict[str, int]:
    usage = (response_data or {}).get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def _content_summary(content: Any) -> dict[str, int]:
    if isinstance(content, str):
        return {"text_chars": len(content), "image_count": 0}
    if not isinstance(content, list):
        return {"text_chars": len(str(content or "")), "image_count": 0}

    text_chars = 0
    image_count = 0
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text_chars += len(str(item.get("text") or ""))
        elif item.get("type") == "image_url":
            image_count += 1
    return {"text_chars": text_chars, "image_count": image_count}


def request_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload or {}
    messages = payload.get("messages") or []
    summary = {
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "text_chars": 0,
        "image_count": 0,
        "tool_schema_count": len(payload.get("tools") or []),
        "input_count": 0,
    }

    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                item = _content_summary(message.get("content"))
                summary["text_chars"] += item["text_chars"]
                summary["image_count"] += item["image_count"]

    inputs = payload.get("input")
    if isinstance(inputs, list):
        summary["input_count"] = len(inputs)
        summary["text_chars"] += sum(len(str(item or "")) for item in inputs)
    elif inputs is not None:
        summary["input_count"] = 1
        summary["text_chars"] += len(str(inputs or ""))
    return summary


def log_api_usage(
    *,
    source: str,
    model: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
    response_data: dict[str, Any] | None = None,
    status_code: int | None = None,
    elapsed_seconds: float | None = None,
    success: bool = True,
    error: str | None = None,
    attempt: int | None = None,
) -> None:
    entry = {
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "model": model,
        "endpoint": endpoint,
        "status_code": status_code,
        "elapsed_seconds": elapsed_seconds,
        "success": success,
        "error": error,
        "attempt": attempt,
        "request_summary": request_summary(payload),
        **_usage(response_data),
    }
    with _LOCK:
        with usage_log_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
