"""Helpers for parsing structured JSON returned by LLM calls."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def _extract_json_text(text: str, *, allow_array: bool = True) -> str:
    text = str(text or "").strip()
    if not text:
        return ""

    shape = r"\{.*?\}|\[.*?\]" if allow_array else r"\{.*?\}"
    fenced = re.search(rf"```(?:json)?\s*({shape})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if not allow_array:
        starts = [text.find("{")] if text.find("{") >= 0 else []
    if not starts:
        return text
    return text[min(starts):]


def parse_llm_json(text: str, *, allow_array: bool = True) -> Any:
    """Parse the first JSON object or list from an LLM response."""
    payload = _extract_json_text(text, allow_array=allow_array)
    decoder = json.JSONDecoder()
    try:
        data, _ = decoder.raw_decode(payload)
        return data
    except json.JSONDecodeError:
        repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", payload)
        data, _ = decoder.raw_decode(repaired)
        return data


def parse_llm_json_object(text: str) -> Dict[str, Any]:
    data = parse_llm_json(text, allow_array=True)
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return {}


def parse_llm_json_list(text: str) -> List[Any]:
    data = parse_llm_json(text, allow_array=True)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []
